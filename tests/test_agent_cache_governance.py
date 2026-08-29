"""Unit tests for api.agent_cache_governance (memory-pressure + idle-TTL valves).

Run: HERMES_WEBUI_PYTHON=/usr/local/lib/hermes-agent/venv/bin/python3.11 \
     python -m pytest tests/test_agent_cache_governance.py -v
"""
import sys
import threading
import time
from types import SimpleNamespace

sys.path.insert(0, ".")

from api.agent_cache_governance import (
    AgentCacheGovernor,
    _positive_int,
    plan_pressure_evictions,
    resolve_memory_high_mb,
    soft_release_transcript,
)


class _FakeAgent:
    def __init__(self, last_activity_ts=None):
        self._last_activity_ts = last_activity_ts
        self._session_messages = [{"role": "user", "content": "x" * 1000}]
        self._db_flush_scan_prefix = [{"role": "user", "content": "y" * 1000}]


def _cache_with(entries):
    """entries: list[(key, agent)] — oldest first (LRU front)."""
    c = {}
    for k, a in entries:
        c[k] = (a, "sig")
    return c


def _governor(cache, **kw):
    return AgentCacheGovernor(cache, threading.Lock(), **kw)


# ── resolve_memory_high_mb ────────────────────────────────────────────────
def test_memory_high_auto_with_total_ram():
    assert resolve_memory_high_mb("auto", total_ram_mb=2000) == 1300
    assert resolve_memory_high_mb("auto", total_ram_mb=512) >= 512  # floor


def test_memory_high_number():
    assert resolve_memory_high_mb("512", total_ram_mb=2000) == 512
    assert resolve_memory_high_mb(400, total_ram_mb=2000) == 400


def test_memory_high_disabled():
    assert resolve_memory_high_mb("0") is None
    assert resolve_memory_high_mb("off") is None
    assert resolve_memory_high_mb("") is None
    assert resolve_memory_high_mb(None) is None
    assert resolve_memory_high_mb("abc") is None


# ── plan_pressure_evictions ───────────────────────────────────────────────
def test_plan_evicts_lru_skips_recent():
    agents = [(f"k{i}", _FakeAgent()) for i in range(10)]
    plan = plan_pressure_evictions(agents, lambda k, a: True, protect_recent=3)
    keys = [k for k, _ in plan]
    assert keys == [f"k{i}" for i in range(7)]  # last 3 protected


def test_plan_never_evicts_midrun():
    agents = [(f"k{i}", _FakeAgent()) for i in range(10)]
    plan = plan_pressure_evictions(
        agents, lambda k, a: k != "k1", protect_recent=0
    )
    keys = [k for k, _ in plan]
    assert "k1" not in keys


def test_plan_max_evictions():
    agents = [(f"k{i}", _FakeAgent()) for i in range(50)]
    plan = plan_pressure_evictions(agents, lambda k, a: True, protect_recent=0)
    assert len(plan) <= 16


# ── soft_release_transcript ───────────────────────────────────────────────
def test_soft_release_drops_transcript_keeps_agent():
    a = _FakeAgent()
    soft_release_transcript(a)
    assert a._session_messages == []
    assert a._db_flush_scan_prefix is None


# ── idle TTL sweep ─────────────────────────────────────────────────────────
def test_idle_sweep_evicts_stale_keeps_fresh():
    now = time.time()
    cache = _cache_with([
        ("stale", _FakeAgent(last_activity_ts=now - 7200)),
        ("fresh", _FakeAgent(last_activity_ts=now - 10)),
    ])
    closed = []

    def close(key, agent):
        closed.append(key)

    g = _governor(cache, idle_ttl_secs=3600, close_agent_fn=close)
    assert g.sweep_idle(now=now) == 1
    assert closed == ["stale"]
    assert "stale" not in cache
    assert "fresh" in cache


def test_idle_sweep_skips_running():
    now = time.time()
    cache = _cache_with([
        ("running", _FakeAgent(last_activity_ts=now - 7200)),
    ])
    g = _governor(
        cache, idle_ttl_secs=3600,
        running_check=lambda k: k == "running",
        close_agent_fn=lambda k, a: None,
    )
    assert g.sweep_idle(now=now) == 0
    assert "running" in cache


def test_idle_sweep_disabled_when_ttl_zero():
    cache = _cache_with([("k", _FakeAgent(last_activity_ts=0))])
    g = _governor(cache, idle_ttl_secs=0, close_agent_fn=lambda k, a: None)
    assert g.sweep_idle(now=time.time()) == 0


# ── memory pressure sweep ──────────────────────────────────────────────────
def test_pressure_sweep_soft_evicts_lru():
    cache = _cache_with([(f"k{i}", _FakeAgent()) for i in range(5)])
    g = _governor(
        cache,
        memory_high_mb=100,
        protect_recent=1,
        running_check=lambda k: False,
    )
    dropped = g.sweep_pressure(rss_mb=500)
    assert dropped == 4  # k0..k3 dropped, k4 protected
    # agents stay in cache (soft) but transcripts are gone
    assert set(cache.keys()) == {"k0", "k1", "k2", "k3", "k4"}
    assert cache["k0"][0]._session_messages == []
    assert cache["k4"][0]._session_messages  # protected: transcript intact


def test_pressure_sweep_noop_below_budget():
    cache = _cache_with([("k", _FakeAgent())])
    g = _governor(cache, memory_high_mb=100, protect_recent=0)
    assert g.sweep_pressure(rss_mb=50) == 0
    assert cache["k"][0]._session_messages


def test_pressure_sweep_skips_running():
    cache = _cache_with([("busy", _FakeAgent())])
    g = _governor(
        cache, memory_high_mb=100, protect_recent=0,
        running_check=lambda k: k == "busy",
    )
    assert g.sweep_pressure(rss_mb=500) == 0
    assert cache["busy"][0]._session_messages


def test_pressure_sweep_disabled_when_budget_none():
    cache = _cache_with([("k", _FakeAgent())])
    g = _governor(cache, memory_high_mb=None, protect_recent=0)
    assert g.sweep_pressure(rss_mb=10 ** 9) == 0


def test_positive_int_parses():
    assert _positive_int("10", 1) == 10
    assert _positive_int("0", 1) == 0  # 0 allowed = disabled
    assert _positive_int("x", 1) == 1
    assert _positive_int(None, 1) == 1
