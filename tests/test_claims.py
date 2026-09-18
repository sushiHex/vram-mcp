"""Tests for vram_mcp.claims — real temp-dir file I/O (atomicity is the point)."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from vram_mcp import _util, claims

# Shared test epoch: every test pins the clock here (in the past relative to
# wall time — which is exactly why write paths must honor now_fn, never the
# real clock, or wall-clock pruning would eat these fixtures).
_T0 = datetime(2026, 7, 13, 18, 0, 0, tzinfo=timezone.utc)


class _Clock:
    """A controllable now_fn: starts at `start`, advances via .tick(seconds)."""

    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def tick(self, seconds: int) -> None:
        self.now += timedelta(seconds=seconds)


def _clock(start: datetime) -> _Clock:
    return _Clock(start)


def test_claim_creates_and_list_claims_returns_it(tmp_path):
    path = tmp_path / "claims.json"
    now_fn = _clock(_T0)
    result = claims.claim("llama3.2", "project-a", "narration",
                          ttl_seconds=3600, path=path, now_fn=now_fn)
    assert "claim_id" in result
    assert result["expires_at"] == "2026-07-13T19:00:00Z"

    active = claims.list_claims(path=path, now_fn=now_fn)
    assert len(active) == 1
    assert active[0]["model"] == "llama3.2:latest"
    assert active[0]["owner"] == "project-a"
    assert active[0]["purpose"] == "narration"


def test_list_claims_filters_by_model(tmp_path):
    path = tmp_path / "claims.json"
    now_fn = _clock(_T0)
    claims.claim("llama3.2", "a", "x", path=path, now_fn=now_fn)
    claims.claim("qwen3:8b", "b", "y", path=path, now_fn=now_fn)

    assert len(claims.list_claims(path=path, now_fn=now_fn)) == 2
    only_qwen = claims.list_claims("qwen3:8b", path=path, now_fn=now_fn)
    assert len(only_qwen) == 1
    assert only_qwen[0]["owner"] == "b"


def test_claim_expires_after_ttl(tmp_path):
    path = tmp_path / "claims.json"
    now_fn = _clock(_T0)
    claims.claim("llama3.2", "a", "x", ttl_seconds=60, path=path, now_fn=now_fn)

    now_fn.tick(30)
    assert len(claims.list_claims(path=path, now_fn=now_fn)) == 1

    now_fn.tick(31)  # 61s total -> past the 60s ttl
    assert claims.list_claims(path=path, now_fn=now_fn) == []


def test_renew_extends_expiry_with_original_ttl(tmp_path):
    path = tmp_path / "claims.json"
    now_fn = _clock(_T0)
    created = claims.claim("llama3.2", "a", "x", ttl_seconds=60, path=path, now_fn=now_fn)

    now_fn.tick(50)
    result = claims.renew(created["claim_id"], path=path, now_fn=now_fn)
    assert result["ok"] is True
    assert result["expires_at"] == "2026-07-13T18:01:50Z"  # now(18:00:50) + 60s

    now_fn.tick(55)  # 105s since creation, 55s since renew -> still active
    assert len(claims.list_claims(path=path, now_fn=now_fn)) == 1


def test_renew_with_new_ttl_overrides(tmp_path):
    path = tmp_path / "claims.json"
    now_fn = _clock(_T0)
    created = claims.claim("llama3.2", "a", "x", ttl_seconds=60, path=path, now_fn=now_fn)
    result = claims.renew(created["claim_id"], ttl_seconds=7200, path=path, now_fn=now_fn)
    assert result["expires_at"] == "2026-07-13T20:00:00Z"


def test_renew_unknown_claim_id_returns_not_ok(tmp_path):
    path = tmp_path / "claims.json"
    assert claims.renew("nonexistent-id", path=path) == {"ok": False}


def test_release_removes_claim(tmp_path):
    path = tmp_path / "claims.json"
    now_fn = _clock(_T0)
    created = claims.claim("llama3.2", "a", "x", path=path, now_fn=now_fn)
    assert claims.release(created["claim_id"], path=path) == {"ok": True}
    assert claims.list_claims(path=path, now_fn=now_fn) == []


def test_release_unknown_claim_id_returns_not_ok(tmp_path):
    path = tmp_path / "claims.json"
    assert claims.release("nonexistent", path=path) == {"ok": False}


def test_multiple_claims_same_model_independent(tmp_path):
    path = tmp_path / "claims.json"
    now_fn = _clock(_T0)
    claims.claim("llama3.2", "session-a", "reason-a", path=path, now_fn=now_fn)
    claims.claim("llama3.2", "session-b", "reason-b", path=path, now_fn=now_fn)
    active = claims.list_claims("llama3.2", path=path, now_fn=now_fn)
    assert {c["owner"] for c in active} == {"session-a", "session-b"}


def test_sequential_claims_both_persist(tmp_path):
    """Each claim()/release() call round-trips through the lock cleanly --
    a stale lock from a prior call never blocks the next one."""
    path = tmp_path / "claims.json"
    now_fn = _clock(_T0)
    claims.claim("a", "x", "p", path=path, now_fn=now_fn)
    claims.claim("b", "y", "p", path=path, now_fn=now_fn)
    assert len(claims.list_claims(path=path, now_fn=now_fn)) == 2


# --- _save retries os.replace on Windows sharing violations


def test_save_retries_replace_on_permission_error_then_succeeds(tmp_path, monkeypatch):
    import os
    real_replace = os.replace
    calls = {"n": 0}

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise PermissionError(13, "The process cannot access the file")
        return real_replace(src, dst)

    monkeypatch.setattr(_util.os, "replace", flaky_replace)
    now_fn = _clock(_T0)
    result = claims.claim("llama3.2", "a", "x", path=tmp_path / "claims.json",
                          now_fn=now_fn)
    assert "claim_id" in result
    assert calls["n"] == 4
    active = claims.list_claims(path=tmp_path / "claims.json", now_fn=now_fn)
    assert len(active) == 1


def test_save_gives_up_after_retries_and_cleans_tmp(tmp_path, monkeypatch):
    path = tmp_path / "claims.json"

    def always_fails(src, dst):
        raise PermissionError(13, "The process cannot access the file")

    monkeypatch.setattr(_util.os, "replace", always_fails)
    now_fn = _clock(_T0)
    with pytest.raises(PermissionError):
        claims.claim("llama3.2", "a", "x", path=path, now_fn=now_fn)
    assert not path.with_suffix(path.suffix + ".tmp").exists()


# --- FIX 3: malformed records are tolerated, not fatal


def test_malformed_records_are_skipped_not_fatal(tmp_path):
    import json
    path = tmp_path / "claims.json"
    good = {
        "claim_id": "good", "model": "llama3.2", "owner": "a", "purpose": "x",
        "claimed_at": "2026-07-13T18:00:00Z", "renewed_at": "2026-07-13T18:00:00Z",
        "ttl_seconds": 3600, "expires_at": "2026-07-13T19:00:00Z",
    }
    bad_missing_key = {"claim_id": "b1", "model": "m"}  # no expires_at
    bad_unparsable = {"claim_id": "b2", "expires_at": "not-a-date"}
    bad_naive_dt = {"claim_id": "b3", "expires_at": "2026-07-13T19:00:00"}  # no tz
    bad_not_a_dict = "garbage"
    path.write_text(json.dumps({"claims": [
        good, bad_missing_key, bad_unparsable, bad_naive_dt, bad_not_a_dict,
    ]}), encoding="utf-8")

    now_fn = _clock(_T0)
    active = claims.list_claims(path=path, now_fn=now_fn)  # must not raise
    assert [r["claim_id"] for r in active] == ["good"]

    # Writes also survive: malformed records get pruned, good one kept.
    claims.claim("qwen3:8b", "b", "y", path=path, now_fn=now_fn)
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert len(on_disk["claims"]) == 2
    assert {r["model"] for r in on_disk["claims"]} == {"llama3.2", "qwen3:8b"}


# --- corrupt/wrong-shape ledgers fail closed


@pytest.mark.parametrize("content", ["[]", "{}", "{{{not json", '{"claims": 42}'])
def test_corrupt_file_raises_without_quarantining_or_overwriting(tmp_path, content):
    path = tmp_path / "claims.json"
    corrupt_path = path.with_suffix(path.suffix + ".corrupt")
    path.write_text(content, encoding="utf-8")

    now_fn = _clock(_T0)
    with pytest.raises(ValueError):
        claims.list_claims(path=path, now_fn=now_fn)
    assert path.read_text(encoding="utf-8") == content
    assert not corrupt_path.exists()


def test_missing_file_is_not_treated_as_corrupt(tmp_path):
    path = tmp_path / "claims.json"
    now_fn = _clock(_T0)
    assert claims.list_claims(path=path, now_fn=now_fn) == []
    assert not path.with_suffix(path.suffix + ".corrupt").exists()


# --- FIX 5: renew doesn't resurrect expired claims; writes prune the file


def test_renew_of_expired_claim_returns_not_ok(tmp_path):
    path = tmp_path / "claims.json"
    now_fn = _clock(_T0)
    created = claims.claim("llama3.2", "a", "x", ttl_seconds=60,
                           path=path, now_fn=now_fn)
    now_fn.tick(61)  # past the TTL
    assert claims.renew(created["claim_id"], path=path, now_fn=now_fn) == {"ok": False}
    assert claims.list_claims(path=path, now_fn=now_fn) == []


def test_expired_records_are_pruned_by_next_write(tmp_path):
    import json
    path = tmp_path / "claims.json"
    now_fn = _clock(_T0)
    claims.claim("old-model", "a", "x", ttl_seconds=60, path=path, now_fn=now_fn)
    now_fn.tick(61)  # first claim expires
    claims.claim("new-model", "b", "y", ttl_seconds=60, path=path, now_fn=now_fn)

    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert len(on_disk["claims"]) == 1  # expired record physically removed
    assert on_disk["claims"][0]["model"] == "new-model:latest"


def test_renew_prunes_expired_records_but_keeps_renewed_one(tmp_path):
    import json
    path = tmp_path / "claims.json"
    now_fn = _clock(_T0)
    claims.claim("doomed", "a", "x", ttl_seconds=60, path=path, now_fn=now_fn)
    keeper = claims.claim("keeper", "b", "y", ttl_seconds=7200,
                          path=path, now_fn=now_fn)
    now_fn.tick(61)  # "doomed" expires, "keeper" still active
    result = claims.renew(keeper["claim_id"], path=path, now_fn=now_fn)
    assert result["ok"] is True

    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert [r["model"] for r in on_disk["claims"]] == ["keeper:latest"]


def test_release_prunes_with_injected_clock_sibling_survives(tmp_path):
    """release() must prune with now_fn, not the wall clock: the test epoch is
    in the past relative to real time, so wall-clock pruning would silently
    destroy the surviving sibling claim."""
    path = tmp_path / "claims.json"
    now_fn = _clock(_T0)
    keeper = claims.claim("llama3.2", "keeper", "still-working",
                          ttl_seconds=3600, path=path, now_fn=now_fn)
    goner = claims.claim("qwen3:8b", "goner", "done", path=path, now_fn=now_fn)

    assert claims.release(goner["claim_id"], path=path, now_fn=now_fn) == {"ok": True}

    survivors = claims.list_claims(path=path, now_fn=now_fn)
    assert [c["claim_id"] for c in survivors] == [keeper["claim_id"]]


# --- reservations: a claim on GB of VRAM rather than on a named model


def test_reserve_round_trip(tmp_path):
    path = tmp_path / "claims.json"
    now = datetime(2026, 7, 19, 12, 0, 0, tzinfo=timezone.utc)
    r = claims.reserve(8.0, "trainer", "dpo run", 3600,
                       pid=1234, path=path, now_fn=lambda: now)
    assert r["claim_id"]
    (rec,) = claims.list_claims(path=path, now_fn=lambda: now)
    assert rec["kind"] == "reservation"
    assert rec["gb"] == 8.0
    assert rec["pid"] == 1234
    assert rec["model"] is None


@pytest.mark.parametrize("gb", [0, -8.0, "eight", None])
def test_reserve_rejects_non_positive_gb(tmp_path, gb):
    """A negative reservation would subtract from another session's total, so
    one session could silently cancel another's. Reject it at the boundary and
    never let the record reach the shared ledger."""
    path = tmp_path / "claims.json"
    now = datetime(2026, 7, 19, 12, 0, 0, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        claims.reserve(gb, "sneaky", "cancel yours", 3600,
                       path=path, now_fn=lambda: now)
    assert claims.list_claims(path=path, now_fn=lambda: now) == []


def test_reservation_expires_by_ttl(tmp_path):
    path = tmp_path / "claims.json"
    now = datetime(2026, 7, 19, 12, 0, 0, tzinfo=timezone.utc)
    claims.reserve(8.0, "trainer", "dpo", 60, path=path, now_fn=lambda: now)
    later = now + timedelta(seconds=61)
    assert claims.list_claims(path=path, now_fn=lambda: later) == []


def test_claim_records_are_tagged_model(tmp_path):
    path = tmp_path / "claims.json"
    now = datetime(2026, 7, 19, 12, 0, 0, tzinfo=timezone.utc)
    claims.claim("llama3", "me", "chat", 3600, path=path, now_fn=lambda: now)
    (rec,) = claims.list_claims(path=path, now_fn=lambda: now)
    assert rec["kind"] == "model"


def test_list_claims_by_model_excludes_reservations(tmp_path):
    path = tmp_path / "claims.json"
    now = datetime(2026, 7, 19, 12, 0, 0, tzinfo=timezone.utc)
    claims.claim("llama3", "me", "chat", 3600, path=path, now_fn=lambda: now)
    claims.reserve(8.0, "trainer", "dpo", 3600, path=path, now_fn=lambda: now)
    got = claims.list_claims("llama3", path=path, now_fn=lambda: now)
    assert len(got) == 1
    assert got[0]["kind"] == "model"


def test_legacy_record_without_kind_still_lists(tmp_path):
    path = tmp_path / "claims.json"
    now = datetime(2026, 7, 19, 12, 0, 0, tzinfo=timezone.utc)
    expires = (now + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    path.write_text(json.dumps({"claims": [
        {"claim_id": "old", "model": "llama3", "owner": "me",
         "purpose": "chat", "ttl_seconds": 3600, "expires_at": expires},
    ]}), encoding="utf-8")
    got = claims.list_claims("llama3", path=path, now_fn=lambda: now)
    assert len(got) == 1


def test_reservation_can_be_released(tmp_path):
    path = tmp_path / "claims.json"
    now = datetime(2026, 7, 19, 12, 0, 0, tzinfo=timezone.utc)
    r = claims.reserve(8.0, "trainer", "dpo", 3600, path=path, now_fn=lambda: now)
    assert claims.release(r["claim_id"], path=path, now_fn=lambda: now)["ok"]
    assert claims.list_claims(path=path, now_fn=lambda: now) == []


def test_model_claims_canonicalize_bare_names_and_registry_ports(tmp_path):
    now = lambda: _T0
    path = tmp_path / "claims.json"
    claims.claim("  llama3.2  ", "owner", "work", path=path, now_fn=now)
    claims.claim("localhost:5000/library/foo", "owner", "work", path=path, now_fn=now)
    assert [r["model"] for r in claims.list_claims(path=path, now_fn=now)] == [
        "llama3.2:latest", "localhost:5000/library/foo:latest",
    ]
    assert len(claims.list_claims("llama3.2", path=path, now_fn=now)) == 1
    assert len(claims.list_claims("localhost:5000/library/foo", path=path, now_fn=now)) == 1


@pytest.mark.parametrize("alias", ["llama3", "library/llama3:latest",
                                   "registry.ollama.ai/library/llama3",
                                   "https://registry.ollama.ai/library/llama3:latest"])
def test_official_model_aliases_share_one_protection_key(tmp_path, alias):
    path = tmp_path / "claims.json"
    claims.claim(alias, "owner", "purpose", path=path, now_fn=lambda: _T0)
    result = claims.begin_operation("REGISTRY.OLLAMA.AI/library/LLAMA3", "unload",
                                    path=path, now_fn=lambda: _T0)
    assert result["reason"] == "model_claimed"
    assert result["claims"][0]["model"] == "llama3:latest"


@pytest.mark.parametrize("model", ["model:", "library//model", "https://registry.ollama.ai/"])
def test_claim_rejects_malformed_model_alias(tmp_path, model):
    with pytest.raises(ValueError):
        claims.claim(model, "owner", "purpose", path=tmp_path / "claims.json", now_fn=lambda: _T0)


@pytest.mark.parametrize("model, owner, purpose", [
    ("", "owner", "purpose"), ("model", " ", "purpose"),
    ("model", "owner", "\t"),
])
def test_claim_rejects_blank_coordination_identity(tmp_path, model, owner, purpose):
    with pytest.raises(ValueError):
        claims.claim(model, owner, purpose, path=tmp_path / "claims.json", now_fn=lambda: _T0)


@pytest.mark.parametrize("ttl", [0, -1, float("inf"), float("nan"), 0.5, 10 ** 20])
def test_claim_and_reservation_reject_invalid_ttl(tmp_path, ttl):
    path = tmp_path / "claims.json"
    with pytest.raises(ValueError):
        claims.claim("model", "owner", "purpose", ttl, path=path, now_fn=lambda: _T0)
    with pytest.raises(ValueError):
        claims.reserve(1, "owner", "purpose", ttl, path=path, now_fn=lambda: _T0)


@pytest.mark.parametrize("gb", [float("inf"), float("-inf"), float("nan"), True, 10 ** 1000])
def test_reserve_rejects_nonfinite_gb(tmp_path, gb):
    with pytest.raises(ValueError):
        claims.reserve(gb, "owner", "purpose", path=tmp_path / "claims.json", now_fn=lambda: _T0)


def test_renew_rejects_invalid_ttl_without_expiring_live_claim(tmp_path):
    path = tmp_path / "claims.json"
    created = claims.claim("model", "owner", "purpose", path=path, now_fn=lambda: _T0)
    with pytest.raises(ValueError):
        claims.renew(created["claim_id"], -1, path=path, now_fn=lambda: _T0)
    assert len(claims.list_claims(path=path, now_fn=lambda: _T0)) == 1


def test_begin_operation_refuses_claimed_model_and_returns_claim_detail(tmp_path):
    path = tmp_path / "claims.json"
    claims.claim("model", "owner", "purpose", path=path, now_fn=lambda: _T0)
    result = claims.begin_operation("model", "unload", path=path, now_fn=lambda: _T0)
    assert result["ok"] is False
    assert result["reason"] == "model_claimed"
    assert result["claims"][0]["model"] == "model:latest"


def test_malformed_active_claim_still_blocks_and_is_visible(tmp_path):
    path = tmp_path / "claims.json"
    path.write_text(json.dumps({"claims": [{
        "claim_id": "claim", "model": "model:latest", "owner": "owner",
        "purpose": "purpose", "claimed_at": "2026-07-13T18:00:00Z",
        "renewed_at": "2026-07-13T18:00:00Z", "ttl_seconds": 3600.0,
        "expires_at": "2026-07-13T19:00:00Z",
    }]}), encoding="utf-8")

    listed = claims.list_coordination(path=path, now_fn=lambda: _T0)
    assert listed["claims"][0]["claim_id"] == "claim"
    assert listed["claims"][0]["ttl_seconds"] is None
    refused = claims.begin_operation("model", "unload", path=path, now_fn=lambda: _T0)
    assert refused["reason"] == "model_claimed"
    assert refused["claims"][0]["claim_id"] == "claim"


def test_pending_operation_refuses_claim_and_same_model_force_operation(tmp_path):
    path = tmp_path / "claims.json"
    started = claims.begin_operation("model", "unload", force=True, path=path, now_fn=lambda: _T0)
    try:
        with pytest.raises(ValueError, match="operation pending"):
            claims.claim("model", "owner", "purpose", path=path, now_fn=lambda: _T0)
        refused = claims.begin_operation("model", "unload", force=True, path=path, now_fn=lambda: _T0)
        assert refused["reason"] == "operation_pending"
    finally:
        assert claims.finish_operation(started["operation_id"], path=path, now_fn=lambda: _T0)["ok"]


def test_live_operation_owner_prevents_expired_lease_from_admitting_race(tmp_path):
    path = tmp_path / "claims.json"
    clock = _clock(_T0)
    started = claims.begin_operation("model", "unload", force=True, path=path, now_fn=clock)
    try:
        clock.tick(claims._OPERATION_LEASE_SECONDS + 1)
        refused = claims.begin_operation("model", "unload", force=True, path=path, now_fn=clock)
        assert refused["reason"] == "operation_pending"
    finally:
        claims.finish_operation(started["operation_id"], path=path, now_fn=clock)
    next_started = claims.begin_operation("model", "unload", force=True, path=path, now_fn=clock)
    assert next_started["ok"]
    claims.finish_operation(next_started["operation_id"], path=path, now_fn=clock)


def test_uncertain_operation_releases_owner_lock_but_renews_durable_grace(tmp_path):
    path = tmp_path / "claims.json"
    clock = _clock(_T0)
    started = claims.begin_operation("model", "unload", force=True, path=path, now_fn=clock)
    finished = claims.finish_operation(started["operation_id"], uncertain=True, path=path, now_fn=clock)
    assert finished["ok"] is True
    assert finished["retained"] is True
    assert finished["expires_at"] == "2026-07-13T18:02:00Z"
    refused = claims.begin_operation("model", "unload", force=True, path=path, now_fn=clock)
    assert refused["reason"] == "operation_pending"
    clock.tick(claims._OPERATION_LEASE_SECONDS + 1)
    next_started = claims.begin_operation("model", "unload", force=True, path=path, now_fn=clock)
    assert next_started["ok"]
    claims.finish_operation(next_started["operation_id"], path=path, now_fn=clock)


@pytest.mark.parametrize("operation", ["bad", {"operation_id": "op"},
                                         {"operation_id": "op", "model": "model"}])
def test_malformed_existing_operation_is_discarded_without_losing_claims(tmp_path, operation):
    path = tmp_path / "claims.json"
    valid_claim = {
        "claim_id": "claim", "model": "model:latest", "owner": "owner",
        "purpose": "purpose", "claimed_at": "2026-07-13T18:00:00Z",
        "renewed_at": "2026-07-13T18:00:00Z", "ttl_seconds": 3600,
        "expires_at": "2026-07-13T19:00:00Z",
    }
    path.write_text(json.dumps({"claims": [valid_claim], "operations": [operation]}), encoding="utf-8")
    assert [r["claim_id"] for r in claims.list_claims(path=path, now_fn=lambda: _T0)] == ["claim"]
    started = claims.begin_operation("model", "unload", force=True, path=path, now_fn=lambda: _T0)
    claims.finish_operation(started["operation_id"], path=path, now_fn=lambda: _T0)
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["operations"] == []
    assert [r["claim_id"] for r in on_disk["claims"]] == ["claim"]


# A record written by a LATER protocol version: identity and lease are readable,
# but its lifecycle, outcome and retry vocabulary are ones this version lacks.
_FOREIGN_OPERATION = {
    "operation_id": "future-op", "model": "model:latest", "kind": "unload",
    "scope": "gpu:index=0", "started_at": "2026-07-13T17:59:00Z",
    "expires_at": "2026-07-13T19:00:00Z",
    "lifecycle": "draining", "outcome": "partially_evicted",
    "reason": {"structured": "not a string"}, "retry_count": "three",
    "some_future_field": {"nested": 1},
}


def _ledger_with_foreign_operation(path):
    path.write_text(json.dumps({"claims": [], "operations": [_FOREIGN_OPERATION]}),
                    encoding="utf-8")
    return path


def test_foreign_operation_survives_a_load_mutate_save_cycle(tmp_path):
    """Erasing another version's in-flight lease is the one thing a shared
    ledger must never do: the writer is still holding that model."""
    path = _ledger_with_foreign_operation(tmp_path / "claims.json")
    started = claims.begin_operation("other:model", "unload", force=True,
                                     path=path, now_fn=lambda: _T0)
    claims.finish_operation(started["operation_id"], path=path, now_fn=lambda: _T0)
    on_disk = json.loads(path.read_text(encoding="utf-8"))["operations"]
    assert [r["operation_id"] for r in on_disk] == ["future-op"]
    assert on_disk[0] == _FOREIGN_OPERATION      # byte-for-byte, fields included


def test_foreign_operation_still_blocks_its_model(tmp_path):
    path = _ledger_with_foreign_operation(tmp_path / "claims.json")
    refused = claims.begin_operation("model:latest", "unload", force=True,
                                     path=path, now_fn=lambda: _T0)
    assert refused["ok"] is False
    assert refused["reason"] == "operation_pending"
    # Refusing without naming the holder leaves a caller no way to recover.
    assert [op["operation_id"] for op in refused["operations"]] == ["future-op"]


def test_foreign_operation_is_reported_without_being_interpreted(tmp_path):
    """Identity, scope and lease are facts we can read; lifecycle and outcome
    are a vocabulary we do not have, so they are null rather than guessed."""
    path = _ledger_with_foreign_operation(tmp_path / "claims.json")
    (op,) = claims.list_coordination(path=path, now_fn=lambda: _T0)["operations"]
    assert op["lifecycle"] == "unrecognized"
    assert (op["operation_id"], op["model"], op["kind"], op["scope"]) == (
        "future-op", "model:latest", "unload", "gpu:index=0")
    assert op["expires_at"] == "2026-07-13T19:00:00Z"
    assert op["lease_expired"] is False
    assert (op["outcome"], op["reason"], op["retry_after"]) == (None, None, None)
    assert op["retry_count"] == 0


def test_foreign_operation_lease_still_expires(tmp_path):
    """Preserving a record is not exempting it: the lease rules are version
    independent, so a lapsed lease with no live owner is still reclaimed."""
    path = _ledger_with_foreign_operation(tmp_path / "claims.json")
    later = datetime(2026, 7, 13, 19, 0, 1, tzinfo=timezone.utc)
    assert claims.list_coordination(path=path, now_fn=lambda: later)["operations"] == []
    started = claims.begin_operation("model:latest", "unload", force=True,
                                     path=path, now_fn=lambda: later)
    assert started["ok"] is True
    claims.finish_operation(started["operation_id"], path=path, now_fn=lambda: later)


@pytest.mark.parametrize("operation", [
    "bad",                                                    # not a record
    {"operation_id": "op"},                                   # no model or lease
    {"operation_id": "", "model": "m", "kind": "unload",      # no identity
     "started_at": "2026-07-13T17:59:00Z",
     "expires_at": "2026-07-13T19:00:00Z"},
    {"operation_id": "op", "model": "m", "kind": "unload",    # unreadable lease
     "started_at": "2026-07-13T17:59:00Z", "expires_at": "not-a-timestamp"},
])
def test_operation_without_identity_or_lease_is_still_discarded(tmp_path, operation):
    """Preserving the uninterpretable does not mean preserving the unusable:
    with no model or no expiry there is nothing left to honour."""
    path = tmp_path / "claims.json"
    path.write_text(json.dumps({"claims": [], "operations": [operation]}),
                    encoding="utf-8")
    assert claims.list_coordination(path=path, now_fn=lambda: _T0)["operations"] == []


def test_warm_operations_serialize_capacity_admission_by_scope(tmp_path):
    path = tmp_path / "claims.json"
    first = claims.begin_operation("one", "warm", force=True, path=path, now_fn=lambda: _T0)
    try:
        refused = claims.begin_operation("two", "warm", force=True, path=path, now_fn=lambda: _T0)
        assert refused["reason"] == "capacity_pending"
        allowed = claims.begin_operation("two", "warm", force=True, path=path,
                                         now_fn=lambda: _T0, scope="gpu:index=1")
        assert allowed["ok"]
        claims.finish_operation(allowed["operation_id"], path=path, now_fn=lambda: _T0)
    finally:
        claims.finish_operation(first["operation_id"], path=path, now_fn=lambda: _T0)


def test_live_aged_warm_owner_still_blocks_same_scope_admission(tmp_path):
    path = tmp_path / "claims.json"
    clock = _clock(_T0)
    first = claims.begin_operation("one", "warm", force=True, path=path, now_fn=clock)
    try:
        clock.tick(claims._OPERATION_LEASE_SECONDS + 1)
        refused = claims.begin_operation("two", "warm", force=True, path=path, now_fn=clock)
        assert refused["reason"] == "capacity_pending"
    finally:
        claims.finish_operation(first["operation_id"], path=path, now_fn=clock)


def test_reservation_waits_for_any_warm_admission_to_resolve(tmp_path):
    path = tmp_path / "claims.json"
    started = claims.begin_operation("model", "warm", force=True, path=path, now_fn=lambda: _T0)
    try:
        with pytest.raises(ValueError, match="warm admission pending"):
            claims.reserve(8, "owner", "purpose", path=path, now_fn=lambda: _T0)
    finally:
        claims.finish_operation(started["operation_id"], path=path, now_fn=lambda: _T0)
    assert claims.reserve(8, "owner", "purpose", path=path, now_fn=lambda: _T0)["claim_id"]


def test_list_coordination_returns_filtered_claims_and_operations(tmp_path):
    path = tmp_path / "claims.json"
    claims.claim("Llama3", "owner", "generation", path=path, now_fn=lambda: _T0)
    claims.reserve(4, "trainer", "lora", path=path, now_fn=lambda: _T0)
    started = claims.begin_operation(
        "llama3:latest", "unload", force=True, scope="gpu:index=1",
        path=path, now_fn=lambda: _T0,
    )
    try:
        state = claims.list_coordination(path=path, now_fn=lambda: _T0)
        assert len(state["claims"]) == 2
        assert len(state["operations"]) == 1
        operation = state["operations"][0]
        assert operation["operation_id"] == started["operation_id"]
        assert operation["model"] == "llama3:latest"
        assert operation["scope"] == "gpu:index=1"
        assert operation["lifecycle"] == "in_flight"
        assert operation["outcome"] is None
        assert operation["retry_count"] == 0
        assert operation["retry_after"] is None

        filtered = claims.list_coordination("LLAMA3", path=path, now_fn=lambda: _T0)
        assert [entry["model"] for entry in filtered["claims"]] == ["llama3:latest"]
        assert [entry["model"] for entry in filtered["operations"]] == ["llama3:latest"]
    finally:
        claims.finish_operation(started["operation_id"], path=path, now_fn=lambda: _T0)


def test_list_coordination_reports_unknown_until_refreshed_expiry(tmp_path):
    path = tmp_path / "claims.json"
    clock = _clock(_T0)
    started = claims.begin_operation("model", "unload", force=True, path=path, now_fn=clock)
    finished = claims.finish_operation(
        started["operation_id"], uncertain=True, reason="request_interrupted",
        path=path, now_fn=clock,
    )
    state = claims.list_coordination(path=path, now_fn=clock)
    [operation] = state["operations"]
    assert operation["lifecycle"] == "unknown"
    assert operation["owner_live"] is False
    assert operation["lease_expired"] is False
    assert operation["outcome"] == "unknown"
    assert operation["reason"] == "request_interrupted"
    assert operation["pending_until"] == finished["expires_at"]
    assert operation["retry_after"] == finished["expires_at"]

    clock.tick(claims._OPERATION_LEASE_SECONDS + 1)
    assert claims.list_coordination(path=path, now_fn=clock)["operations"] == []


def test_list_coordination_defaults_missing_pending_boundary(tmp_path):
    path = tmp_path / "claims.json"
    expires = "2026-07-13T18:02:00Z"
    path.write_text(json.dumps({"claims": [], "operations": [{
        "operation_id": "op", "model": "model:latest", "kind": "unload",
        "started_at": "2026-07-13T18:00:00Z", "expires_at": expires,
        "pending_until": None,
    }]}), encoding="utf-8")

    [operation] = claims.list_coordination(path=path, now_fn=lambda: _T0)["operations"]
    assert operation["pending_until"] == expires


def test_retry_count_survives_expiry_and_list_pruning(tmp_path):
    path = tmp_path / "claims.json"
    clock = _clock(_T0)
    first = claims.begin_operation("model", "unload", force=True, path=path, now_fn=clock)
    claims.finish_operation(first["operation_id"], uncertain=True, path=path, now_fn=clock)
    clock.tick(claims._OPERATION_LEASE_SECONDS + 1)

    # A polling read may prune the old unknown record before the retry starts.
    assert claims.list_coordination(path=path, now_fn=clock)["operations"] == []
    second = claims.begin_operation("model", "unload", force=True, path=path, now_fn=clock)
    try:
        assert second["ok"] is True
        on_disk = json.loads(path.read_text(encoding="utf-8"))
        assert on_disk["operations"][0]["retry_count"] == 1
    finally:
        claims.finish_operation(second["operation_id"], path=path, now_fn=clock)
