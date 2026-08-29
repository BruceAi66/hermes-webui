"""Agent-cache governance for the WebUI's per-session AIAgent cache.

The WebUI keeps one ``AIAgent`` per session in ``SESSION_AGENT_CACHE`` (an
OrderedDict LRU) so cross-turn state (``_user_turn_count``, memory-provider
state) survives between messages.  Each cached agent pins the session's full
live transcript (``_session_messages``) in RAM — tens of MB on a tool-heavy
session — so the LRU *count* cap alone does not bound the process's resident
memory: warm sessions churn, transcripts keep growing, and RSS climbs until an
operator restarts the service.

The gateway solved exactly this problem with three valves
(``agent.agent_cache`` in config_defaults.py, issue #80764): an LRU cap, an
idle TTL, and a memory-pressure pass that soft-evicts LRU transcripts once
anonymous RSS crosses a budget.  This module ports the latter two to the WebUI:

* ``sweep_idle_cached_agents`` — fully evicts cached agents idle past the TTL.
* ``sweep_agent_cache_under_pressure`` — soft-evicts the least-recently-used
  transcripts (drops ``_session_messages`` only, keeping the agent object so
  cross-turn state survives); the transcript is rebuilt from the persisted
  session on the next turn.

Both are driven by one daemon thread started from ``server.py``.  Everything
here is pure/read-only so it can be unit-tested without a running server.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_BYTES_PER_MB = 1024 * 1024
# Fraction of the resolved memory limit at which we start shedding transcripts.
# Deliberately well under the limit so eviction happens while the process still
# has room to breathe (parity with gateway.agent_cache_pressure).
_AUTO_BUDGET_FRACTION = 0.65
# Below this a "budget" is noise — tiny boxes would evict on every pass.
_AUTO_BUDGET_FLOOR_MB = 512
# Upper bound on soft-evictions per pass, so one pressure burst cannot stall
# the server tearing down a dozen agents (parity with the gateway).
_MAX_EVICTIONS_PER_PASS = 16


def _positive_int(value: Any, default: int) -> int:
    """Parse an int >= 0; fall back on anything malformed (0 = disabled)."""
    if isinstance(value, bool) or value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _cgroup_memory_limit_bytes() -> Optional[int]:
    """Return the memory limit this process runs under, if cgroup-capped.

    Mirrors gateway/agent_cache_pressure.py: prefers cgroup v2 ``memory.max``
    (and ``memory.high``) on the process's own cgroup, then the root files.
    ``max`` / near-2^63 sentinels mean "unlimited" → None.
    """
    if not sys_platform_linux():
        return None
    candidates: List[str] = []
    try:
        own = _own_cgroup_path()
        if own and own != "/":
            candidates.extend(
                (
                    f"/sys/fs/cgroup{own}/memory.high",
                    f"/sys/fs/cgroup{own}/memory.max",
                )
            )
    except Exception:
        pass
    candidates.extend(
        (
            "/sys/fs/cgroup/memory.high",
            "/sys/fs/cgroup/memory.max",
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        )
    )
    for candidate in candidates:
        try:
            with open(candidate, "r", encoding="utf-8") as fh:
                raw = fh.read().strip()
        except OSError:
            continue
        if not raw or raw == "max":
            continue
        try:
            limit = int(raw)
        except ValueError:
            continue
        if limit <= 0 or limit >= (1 << 62):
            continue
        return limit
    return None


def sys_platform_linux() -> bool:
    import sys

    return sys.platform == "linux"


def _own_cgroup_path() -> Optional[str]:
    """Read the process's own cgroup path (v2: single line; v1: pick memory)."""
    try:
        with open("/proc/self/cgroup", "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(":", 2)
                if len(parts) == 3 and parts[1] == "memory":
                    return parts[2]
                if len(parts) == 3 and parts[2].startswith("/"):
                    # v2 single hierarchy: controllers field empty.
                    return parts[2]
    except OSError:
        return None
    return None


def read_anon_rss_mb() -> Optional[int]:
    """Return the process's anonymous resident memory in MB, or None.

    Anonymous pages are the ones cached transcripts live in.  Uses
    /proc/self/status RssAnon; falls back to psutil total RSS if unavailable.
    """
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("RssAnon:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) // 1024
    except (OSError, ValueError):
        pass
    try:
        import psutil  # type: ignore

        return int(psutil.Process(os.getpid()).memory_info().rss / _BYTES_PER_MB)
    except Exception:
        return None


def resolve_memory_high_mb(raw: Any, total_ram_mb: Optional[int] = None) -> Optional[int]:
    """Resolve the memory-pressure budget in MB.

    ``raw`` accepts:
      * "auto" — cgroup memory limit × _AUTO_BUDGET_FRACTION (floored at
        _AUTO_BUDGET_FLOOR_MB), falling back to total RAM × fraction.
      * a number — the budget in MB.
      * 0 / "0" / "" / "off" — pressure pass disabled (None).

    ``total_ram_mb`` is injectable for tests; None → read from the system.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = raw.strip()
        if raw == "" or raw.lower() in ("0", "off", "none", "false"):
            return None
        if raw.lower() == "auto":
            limit = _cgroup_memory_limit_bytes()
            if limit is not None:
                budget = int(limit * _AUTO_BUDGET_FRACTION / _BYTES_PER_MB)
                return budget if budget >= _AUTO_BUDGET_FLOOR_MB else _AUTO_BUDGET_FLOOR_MB
            if total_ram_mb is None:
                total_ram_mb = _total_ram_mb()
            if total_ram_mb is None:
                return None
            budget = int(total_ram_mb * _AUTO_BUDGET_FRACTION)
            return budget if budget >= _AUTO_BUDGET_FLOOR_MB else _AUTO_BUDGET_FLOOR_MB
        try:
            value = int(raw)
        except ValueError:
            return None
        if value <= 0:
            return None
        return value
    if isinstance(raw, (int, float)):
        value = int(raw)
        return value if value > 0 else None
    return None


def _total_ram_mb() -> Optional[int]:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) // 1024
    except (OSError, ValueError):
        pass
    try:
        import os as _os

        pages = _os.sysconf("SC_PHYS_PAGES")
        page_size = _os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size / _BYTES_PER_MB)
    except Exception:
        return None


def _agent_last_activity(agent: Any) -> Optional[float]:
    """Return the agent's last-activity timestamp (mirrors gateway semantics).

    Falls back to the attribute the AIAgent maintains on every turn.
    """
    if agent is None:
        return None
    ts = getattr(agent, "_last_activity_ts", None)
    if isinstance(ts, (int, float)):
        return float(ts)
    return None


def plan_pressure_evictions(
    ordered: List[Tuple[str, Any]],
    is_evictable,
    protect_recent: int = 8,
    max_evictions: int = _MAX_EVICTIONS_PER_PASS,
) -> List[Tuple[str, Any]]:
    """Pick LRU sessions to soft-evict under memory pressure.

    ``ordered`` is the cache's LRU order (oldest first).  ``is_evictable(key,
    agent)`` decides whether a session may be shed (e.g. not mid-turn, live
    transcript already on disk).  The ``protect_recent`` most-recently-used
    sessions are never touched — their warm prompt cache is worth the most.
    """
    plan: List[Tuple[str, Any]] = []
    if not ordered:
        return plan
    if protect_recent > 0:
        ordered = ordered[: len(ordered) - protect_recent]
    for key, agent in ordered:
        if len(plan) >= max_evictions:
            break
        try:
            if is_evictable(key, agent):
                plan.append((key, agent))
        except Exception:
            logger.debug("pressure evictability check failed for %s", key, exc_info=True)
    return plan


def soft_release_transcript(agent: Any) -> None:
    """Drop the live transcript from a cached agent without tearing it down.

    The next turn rebuilds messages from the persisted session (the WebUI feeds
    ``get_state_db_session_messages`` fresh each turn), so dropping the in-RAM
    list is safe and is the single biggest lever on resident memory.  Keeps the
    agent object itself so cross-turn state (_user_turn_count, memory-provider
    state) survives the pressure pass.
    """
    if agent is None:
        return
    try:
        if hasattr(agent, "_session_messages"):
            agent._session_messages = []
    except Exception:
        logger.debug("soft-release transcript failed", exc_info=True)
    # _db_flush_scan_prefix is a shallow copy of the flushed transcript — it
    # shares every message dict, so leaving it pins the multi-MB content strings
    # the eviction exists to free (parity with gateway #80764).
    try:
        if hasattr(agent, "_db_flush_scan_prefix"):
            agent._db_flush_scan_prefix = None
    except Exception:
        logger.debug("soft-release flush prefix failed", exc_info=True)


class AgentCacheGovernor:
    """One governance pass: idle TTL eviction + memory-pressure soft eviction.

    Holds only a reference to the cache + lock + config snapshot so it can be
    constructed in-process with zero import cycles (streaming.py owns the
    cache; the governor is pure orchestration).
    """

    def __init__(
        self,
        cache: Dict[str, Any],
        lock: threading.Lock,
        *,
        idle_ttl_secs: int = 3600,
        memory_high_mb: Optional[int] = None,
        protect_recent: int = 8,
        running_check=None,
        close_agent_fn=None,
        eviction_hook=None,
    ) -> None:
        self._cache = cache
        self._lock = lock
        self.idle_ttl_secs = idle_ttl_secs
        self.memory_high_mb = memory_high_mb
        self.protect_recent = protect_recent
        # running_check(key) → bool: True when the session has a live turn.
        self._running_check = running_check or (lambda key: False)
        # close_agent_fn(key, agent) → bool: full teardown of an evicted agent
        # (commit memory, close session DB).  Used by the idle-TTL pass.
        self._close_agent_fn = close_agent_fn
        # eviction_hook(key, agent, soft: bool): observability (counters/logs).
        self._eviction_hook = eviction_hook

    # ── Idle TTL ────────────────────────────────────────────────────────────
    def sweep_idle(self, now: Optional[float] = None) -> int:
        """Fully evict cached agents idle past the TTL (0 = disabled).

        Returns the number of agents evicted.  Runs the close callback outside
        the cache lock so provider I/O never blocks cache users.
        """
        if not self.idle_ttl_secs:
            return 0
        if now is None:
            now = time.time()
        to_evict: List[Tuple[str, Any]] = []
        with self._lock:
            for key, entry in list(self._cache.items()):
                agent = entry[0] if isinstance(entry, tuple) and entry else None
                if agent is None:
                    continue
                if self._running_check(key):
                    continue  # mid-turn — don't tear it down
                last_activity = _agent_last_activity(agent)
                if last_activity is None:
                    continue  # unknown — leave it (parity with gateway)
                if (now - last_activity) > self.idle_ttl_secs:
                    self._cache.pop(key, None)
                    to_evict.append((key, agent))
        for key, agent in to_evict:
            try:
                if self._close_agent_fn is not None:
                    self._close_agent_fn(key, agent)
                if self._eviction_hook is not None:
                    self._eviction_hook(key, agent, soft=False)
                logger.info(
                    "Agent-cache governor: idle evicted session=%s "
                    "(idle_ttl=%ss, cache_size=%d)",
                    key, self.idle_ttl_secs, len(self._cache),
                )
            except Exception:
                logger.debug("idle eviction teardown failed for %s", key, exc_info=True)
        return len(to_evict)

    # ── Memory pressure ─────────────────────────────────────────────────────
    def sweep_pressure(self, rss_mb: Optional[int] = None) -> int:
        """Soft-evict LRU transcripts once anonymous RSS crosses the budget.

        Returns the number of transcripts dropped (0 when memory is fine or the
        pass is disabled).
        """
        if not self.memory_high_mb:
            return 0
        if rss_mb is None:
            rss_mb = read_anon_rss_mb()
        if rss_mb is None or rss_mb < self.memory_high_mb:
            return 0

        def _is_evictable(key: str, agent: Any) -> bool:
            if agent is None:
                return False
            if self._running_check(key):
                return False
            return True

        plan = plan_pressure_evictions(
            [
                (key, entry[0] if isinstance(entry, tuple) and entry else entry)
                for key, entry in self._cache.items()
            ],
            _is_evictable,
            protect_recent=self.protect_recent,
        )
        for key, agent in plan:
            try:
                soft_release_transcript(agent)
                if self._eviction_hook is not None:
                    self._eviction_hook(key, agent, soft=True)
                logger.info(
                    "Agent-cache governor: pressure soft-evicted session=%s "
                    "(rss=%sMB budget=%sMB, cache_size=%d)",
                    key, rss_mb, self.memory_high_mb, len(self._cache),
                )
            except Exception:
                logger.debug("pressure soft-evict failed for %s", key, exc_info=True)
        return len(plan)

    def run_pass(self) -> Dict[str, int]:
        """Run one full governance pass; returns {idle_evicted, pressure_dropped}."""
        idle = self.sweep_idle()
        pressure = self.sweep_pressure()
        return {"idle_evicted": idle, "pressure_dropped": pressure}
