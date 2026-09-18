"""The output schema is the contract, and both directions of it are enforced.

FastMCP validates every tool result against its declared output schema and
silently DROPS any key the schema does not name. A field a producer sets but
`schemas.py` omits therefore never reaches the caller, with no error anywhere —
which is how `free_mb`, `scope` and `coordination_warning` were lost. A field
the schema declares but a producer omits fails the opposite way, loudly, as a
pydantic validation error at call time.

So these tests assert key-set EQUALITY rather than containment, across every
outcome a tool can produce. Equality catches both directions at the point the
result is built, instead of at the point an agent notices something missing.
"""
from typing import get_type_hints

import pytest

pytest.importorskip("mcp")

from vram_mcp import schemas, server  # noqa: E402
from vram_mcp.observations import Observation  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(server._claims, "_DEFAULT_PATH", tmp_path / "claims.json")
    monkeypatch.setattr(server._audit, "log_action", lambda **kwargs: None)


class _Backend:
    """An Ollama stand-in whose residency calls report a chosen outcome."""

    base_url = "http://ollama.test:11434"

    def __init__(self, outcome="succeeded", loaded=(), loaded_known=True):
        self.outcome = outcome
        self.loaded = list(loaded)
        self.loaded_known = loaded_known

    def observe_loaded(self):
        if not self.loaded_known:
            return Observation(None, "ollama:/api/ps", error="transport down")
        return Observation(self.loaded, "ollama:/api/ps")

    def ps(self):
        return self.loaded

    def observe_tags(self):
        return Observation({"big:32b": 20000}, "sizes")

    def change_residency(self, model, _keep_alive, *, resident):
        return {"ok": self.outcome == "succeeded", "outcome": self.outcome,
                "model": model, "detail": f"stub {self.outcome}",
                "resident": resident if self.outcome == "succeeded" else None}


def _use(monkeypatch, outcome="succeeded", free_mb=40000, loaded=(),
         loaded_known=True):
    monkeypatch.setattr(server, "_ollama",
                        _Backend(outcome, loaded, loaded_known))
    monkeypatch.setattr(server, "_gpu_reading", lambda: [{"free_mb": free_mb}])


def _break_finish(monkeypatch):
    """Make lease cleanup fail, the one path that sets coordination_warning."""
    def boom(*args, **kwargs):
        raise OSError("ledger unavailable during cleanup")
    monkeypatch.setattr(server._claims, "finish_operation", boom)


def _pending_warm():
    """Leave a warm holding this GPU scope's admission capacity."""
    server._claims.begin_operation(
        "other:7b", "warm", True, scope=f"gpu:index={server._GPU_INDEX}")


def _release_operation_locks():
    for fd in list(server._claims._OPERATION_LOCKS.values()):
        try:
            server._claims._release_operation_lock(fd)
        except OSError:
            pass
    server._claims._OPERATION_LOCKS.clear()


# (tool, schema, builder) — builder returns the raw impl result for one path.
# Every tool with a declared output schema must appear here; the completeness
# test below fails if one is added without coverage.
CASES = [
    ("claim", schemas.ClaimResult, "succeeded",
     lambda mp: server._claim_impl("m:1", "owner", "purpose", 60)),
    ("claim", schemas.ClaimResult, "ledger_error",
     lambda mp: server._claim_impl("m:1", "   ", "purpose", 60)),

    ("reserve", schemas.ReserveResult, "succeeded",
     lambda mp: server._reserve_impl(1.0, "owner", "purpose", 60, None)),
    ("reserve", schemas.ReserveResult, "ledger_error",
     lambda mp: server._reserve_impl(-8, "owner", "purpose", 60, None)),

    ("renew", schemas.RenewResult, "succeeded",
     lambda mp: server._renew_impl(
         server._claim_impl("m:renew", "o", "p", 600)["claim_id"], 60)),
    ("renew", schemas.RenewResult, "not_active",
     lambda mp: server._renew_impl("no-such-claim", 60)),
    ("renew", schemas.RenewResult, "ledger_error",
     lambda mp: server._renew_impl("   ", 60)),

    ("release", schemas.ReleaseResult, "succeeded",
     lambda mp: server._release_impl(
         server._claim_impl("m:rel", "o", "p", 600)["claim_id"])),
    ("release", schemas.ReleaseResult, "not_found",
     lambda mp: server._release_impl("no-such-claim")),

    ("list_claims", schemas.ListClaimsResult, "succeeded",
     lambda mp: server._list_claims_impl(None)),
    ("list_claims", schemas.ListClaimsResult, "ledger_error",
     lambda mp: server._list_claims_impl("   ")),

    ("unload", schemas.ResidencyResult, "succeeded",
     lambda mp: (_use(mp), server._unload_impl("u:ok", True, "t"))[-1]),
    ("unload", schemas.ResidencyResult, "failed",
     lambda mp: (_use(mp, "failed"), server._unload_impl("u:bad", True, "t"))[-1]),
    ("unload", schemas.ResidencyResult, "unknown",
     lambda mp: (_use(mp, "unknown"), server._unload_impl("u:unk", True, "t"))[-1]),
    ("unload", schemas.ResidencyResult, "refused_operation_pending",
     lambda mp: (_use(mp, "unknown"),
                 server._unload_impl("u:pend", True, "t"),
                 server._unload_impl("u:pend", True, "t"))[-1]),
    ("unload", schemas.ResidencyResult, "refused_model_claimed",
     lambda mp: (_use(mp),
                 server._claim_impl("u:held", "other", "chat", 600),
                 server._unload_impl("u:held", False, "t"))[-1]),
    ("unload", schemas.ResidencyResult, "coordination_error",
     lambda mp: (_use(mp), server._unload_impl("   ", True, "t"))[-1]),
    # Cleanup failure is the only producer of `coordination_warning`, and it is
    # the field whose loss mattered most: it says the lease may still be held.
    ("unload", schemas.ResidencyResult, "cleanup_failed",
     lambda mp: (_use(mp), _break_finish(mp),
                 server._unload_impl("u:cw", True, "t"))[-1]),

    ("warm", schemas.ResidencyResult, "succeeded",
     lambda mp: (_use(mp), server._warm_impl("big:32b", "5m", "t", True))[-1]),
    ("warm", schemas.ResidencyResult, "refused_invalid_keep_alive",
     lambda mp: (_use(mp), server._warm_impl("big:32b", "0", "t", False))[-1]),
    ("warm", schemas.ResidencyResult, "refused_insufficient_headroom",
     lambda mp: (_use(mp, free_mb=8000),
                 server._reserve_impl(7.0, "trainer", "sdxl", 600, None),
                 server._warm_impl("big:32b", "5m", "t", False))[-1]),
    # A warm blocked by another warm is the only producer of `scope`.
    ("warm", schemas.ResidencyResult, "refused_capacity_pending",
     lambda mp: (_use(mp), _pending_warm(), server._warm_impl(
         "big:32b", "5m", "t", True))[-1]),

    ("ensure_free", schemas.EnsureFreeResult, "already_free",
     lambda mp: (_use(mp), server._ensure_free_impl(1, True, "t"))[-1]),
    ("ensure_free", schemas.EnsureFreeResult, "with_attempts",
     lambda mp: (_use(mp, free_mb=100, loaded=[
         {"name": "big:32b", "size": 20 * 1024 ** 3,
          "size_vram": 20 * 1024 ** 3, "expires_at": None}]),
         server._ensure_free_impl(1, True, "t"))[-1]),
    ("ensure_free", schemas.EnsureFreeResult, "coordination_error",
     lambda mp: (_use(mp), server._ensure_free_impl(-1, True, "t"))[-1]),
    ("ensure_free", schemas.EnsureFreeResult, "ollama_residency_unknown",
     lambda mp: (_use(mp, free_mb=100, loaded_known=False),
                 server._ensure_free_impl(1, True, "t"))[-1]),
]


@pytest.mark.parametrize(
    "tool,schema,path,build",
    CASES,
    ids=[f"{tool}-{path}" for tool, _schema, path, _build in CASES],
)
def test_producer_emits_exactly_the_declared_fields(tool, schema, path, build, monkeypatch):
    try:
        result = build(monkeypatch)
    finally:
        _release_operation_locks()
    declared = set(get_type_hints(schema))
    assert set(result) - declared == set(), (
        f"{tool} ({path}) sets fields {schemas.__name__}.{schema.__name__} does not "
        f"declare; FastMCP would drop them silently"
    )
    assert declared - set(result) == set(), (
        f"{tool} ({path}) omits fields {schema.__name__} declares; FastMCP would "
        f"reject the result at call time"
    )


def test_ensure_free_lifts_a_blocked_attempts_retry_boundary(monkeypatch):
    """The batch has no operation of its own, so its boundary comes from the
    attempt that is actually blocked. Reporting null here would tell a caller
    to retry immediately into the same refusal."""
    loaded = [{"name": "big:32b", "size": 20 * 1024 ** 3,
               "size_vram": 20 * 1024 ** 3, "expires_at": None}]
    _use(monkeypatch, "unknown", free_mb=100, loaded=loaded)
    try:
        uncertain = server._ensure_free_impl(1, True, "t")
        blocked = server._ensure_free_impl(1, True, "t")
    finally:
        _release_operation_locks()

    [attempt] = uncertain["attempts"]
    assert attempt["pending_until"], "the uncertain eviction should hold a lease"
    assert uncertain["pending_until"] == attempt["pending_until"]
    assert uncertain["retry_after"] == attempt["pending_until"]

    # The retained lease now refuses the next batch; its boundary must surface.
    [refused] = blocked["attempts"]
    assert refused["reason"] == "operation_pending"
    assert blocked["pending_until"] == refused["pending_until"]


def test_nested_attempts_match_the_residency_schema(monkeypatch):
    """ensure_free embeds eviction attempts, so they carry their own contract."""
    _use(monkeypatch, free_mb=100, loaded=[
        {"name": "big:32b", "size": 20 * 1024 ** 3,
         "size_vram": 20 * 1024 ** 3, "expires_at": None}])
    try:
        result = server._ensure_free_impl(1, True, "t")
    finally:
        _release_operation_locks()
    declared = set(get_type_hints(schemas.ResidencyResult))
    assert result["attempts"], "expected at least one eviction attempt"
    for attempt in result["attempts"]:
        assert set(attempt) == declared


def test_foreign_operation_validates_against_the_published_schema(monkeypatch, tmp_path):
    """A lifecycle value claims.py can emit but schemas.py does not declare
    would be rejected at call time, turning a preserved record into a failed
    tool call. Pin the two together on the path that carries them."""
    import json

    ledger = tmp_path / "claims.json"
    ledger.write_text(json.dumps({"claims": [], "operations": [{
        "operation_id": "future-op", "model": "m:1", "kind": "unload",
        "scope": "gpu:index=0", "started_at": "2026-09-18T00:00:00Z",
        "expires_at": "2099-01-01T00:00:00Z",
        "lifecycle": "draining", "outcome": "partially_evicted",
        "reason": {"structured": "not a string"}, "retry_count": "three",
    }]}), encoding="utf-8")
    monkeypatch.setattr(server._claims, "_DEFAULT_PATH", ledger)

    result = server._list_claims_impl(None)
    (operation,) = result["operations"]
    assert operation["lifecycle"] == "unrecognized"
    tool = server.mcp._tool_manager.get_tool("list_claims")
    tool.fn_metadata.output_model.model_validate(result)


def test_every_schema_bearing_tool_is_covered():
    """A new coordination tool cannot ship without a contract case here."""
    with_schemas = {
        name for name, tool in server.mcp._tool_manager._tools.items()
        if tool.fn_metadata.output_schema is not None
    }
    assert with_schemas == {tool for tool, _s, _p, _b in CASES}
