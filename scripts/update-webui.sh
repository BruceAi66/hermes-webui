#!/usr/bin/env bash
# WebUI 更新流程:拉上游新版本 → 合进本地 master(保留本地修复) → 重启 → 验证
# 用法: sudo ./scripts/update-webui.sh [--dry-run]
set -euo pipefail

WEBUI_DIR="/root/hermes-webui"
BRANCH="master"
REMOTE="origin"        # 上游 nesquena/hermes-webui
# 语法门用的解释器:必须 ≥3.10(代码含 `X | None` 标注),系统 python3 是 3.6
PYBIN="/usr/local/lib/hermes-agent/venv/bin/python3.11"
[[ -x "$PYBIN" ]] || PYBIN="$(command -v python3)"
DRY=""
if [[ "${1:-}" == "--dry-run" ]]; then DRY="--dry-run"; fi

cd "$WEBUI_DIR"

# 陈旧合并保护:上次运行若在冲突中途硬退出,工作区会残留冲突标记(树不可启动)。
# 开跑前先清掉,别把坏树一路带到重启步骤。
if [[ -f .git/MERGE_HEAD ]]; then
    echo "!! 发现未完成的合并,先回滚到本地 HEAD"
    git merge --abort || true
fi

echo "== 1/4 拉取上游 =="
git fetch "$REMOTE" master

# 无更新保护:上游 origin/master 已是本地 HEAD 的祖先(=最新已合入)→ 跳过,不动服务
# 不能比 SHA:本地 master 永远领先上游(带本地修复 commit)
if git merge-base --is-ancestor "$REMOTE/master" HEAD; then
    echo "== 无上游更新(上游最新已合入),跳过 =="
    exit 0
fi
echo "上游有更新,准备合并: $(git rev-parse --short "$REMOTE/master")"

echo "== 2/4 合并上游(保留本地修复) =="
# 本地 master = 上游 + 修复。上游更新时可能触碰 api/config.py / server.py,
# 若冲突需手动解决(修复集中在 agent_cache_governance.py,通常无冲突)
if [[ "$DRY" == "--dry-run" ]]; then
    # 显示我们相对上游领先的东西(=本地修复,更新后应保留)
    git diff --stat "$REMOTE/master"...HEAD
else
    # 冲突必须回滚:set -e 下 merge 返回 1 会直接退出,把工作区留在带冲突
    # 标记的中间态 → 树不可启动,WebUI 一旦重启就再也起不来(2026-09-27 实测:
    # api/streaming.py 冲突留下 MERGE_HEAD + 冲突标记,进程重启即死)。
    if ! git merge "$REMOTE/master" --no-edit; then
        git merge --abort || true
        echo "!! 合并冲突,已回滚(git merge --abort),工作区回到本地 HEAD。" >&2
        echo "   手动解决:cd $WEBUI_DIR && git merge $REMOTE/master" >&2
        exit 1
    fi
fi

echo "== 3/4 推送到自己 fork =="
TOKEN=$(grep '^GITHUB_TOKEN=' /root/.hermes/.env | cut -d= -f2 | tr -d '"' | tr -d "'")
PUSH_URL="https://x-access-token:${TOKEN}@github.com/BruceAi66/hermes-webui.git"
if [[ "$DRY" == "--dry-run" ]]; then
    echo "(dry-run: git push to BruceAi66/hermes-webui)"
else
    git push "$PUSH_URL" "$BRANCH"
    unset TOKEN
fi

echo "== 4/4 重启 + 验证 =="
if [[ "$DRY" == "--dry-run" ]]; then
    echo "(dry-run: systemctl restart hermes-webui)"
else
    # 重启前语法门:树能通过 compileall 才允许重启,否则保留正在跑的旧进程。
    # 半合并/坏文件一旦重启 = WebUI 直接起不来(连界面都没了,只能上 SSH)。
    if ! "$PYBIN" -m compileall -q api server.py bootstrap.py; then
        echo "!! 语法门不通过,放弃重启(继续跑旧进程)。需人工检查。" >&2
        exit 1
    fi
    systemctl restart hermes-webui
    sleep 6
    systemctl is-active hermes-webui
    journalctl -u hermes-webui --since "30 seconds ago" --no-pager | grep -E "governor|Agent-cache" | tail -2 || true
    # ^ 无匹配时 grep 退出码为 1,在 set -e 下会把成功的一轮更新判成 exit 1(每周日假失败,2026-09-15 修)
fi
echo "done"
