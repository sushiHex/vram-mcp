"""Tests for vram_mcp.audit — event log + diff detector, real tmp_path files."""
from datetime import datetime, timedelta, timezone

import pytest

from vram_mcp import audit

_T0 = datetime(2026, 7, 16, 18, 0, 0, tzinfo=timezone.utc)


def _clock(start=_T0):
    st = {"now": start}
    def now_fn():
        return st["now"]
    def tick(s):
        st["now"] = st["now"] + timedelta(seconds=s)
    now_fn.tick = tick
    return now_fn


def test_log_action_writes_typed_event(tmp_path):
    p = tmp_path / "events.jsonl"
    audit.log_action(action="unload", target="qwen3:8b", kind="ollama",
                     actor="retro-repo", force=True, outcome="ok",
                     detail="unloaded", now_fn=_clock(), path=p)
    events = audit.read_events(path=p)
    assert len(events) == 1
    e = events[0]
    assert e["type"] == "action" and e["action"] == "unload"
    assert e["target"] == "qwen3:8b" and e["actor"] == "retro-repo"
    assert e["outcome"] == "succeeded"  # legacy input is normalized on write
    assert e["force"] is True and e["ts"] == "2026-07-16T18:00:00Z"


def test_read_events_newest_first_and_limit(tmp_path):
    p = tmp_path / "events.jsonl"
    clk = _clock()
    for name in ["a", "b", "c"]:
        audit.log_action(action="warm", target=name, kind="ollama",
                         now_fn=clk, path=p)
        clk.tick(1)
    got = audit.read_events(limit=2, path=p)
    assert [e["target"] for e in got] == ["c:latest", "b:latest"]


def test_read_events_filter_by_model_and_type(tmp_path):
    p = tmp_path / "events.jsonl"
    clk = _clock()
    audit.log_action(action="unload", target="qwen3:8b", kind="ollama", now_fn=clk, path=p)
    audit.log_action(action="warm", target="llama3.2", kind="ollama", now_fn=clk, path=p)
    assert [e["target"] for e in audit.read_events(model="qwen3:8b", path=p)] == ["qwen3:8b"]
    assert [e["action"] for e in audit.read_events(type="action", path=p)]  # all are actions


def test_read_events_model_filter_matches_legacy_bare_ollama_name(tmp_path):
    p = tmp_path / "events.jsonl"
    p.write_text(
        '{"ts":"2026-07-16T18:00:00Z","type":"action","kind":"ollama",'
        '"target":"llama3.2","action":"unload","outcome":"ok"}\n',
        encoding="utf-8",
    )
    got = audit.read_events(model="llama3.2:latest", path=p)
    assert [event["target"] for event in got] == ["llama3.2"]


def test_read_events_since_floor(tmp_path):
    p = tmp_path / "events.jsonl"
    clk = _clock()
    audit.log_action(action="warm", target="old", kind="ollama", now_fn=clk, path=p)
    clk.tick(120)
    audit.log_action(action="warm", target="new", kind="ollama", now_fn=clk, path=p)
    got = audit.read_events(since="2026-07-16T18:01:00Z", path=p)
    assert [e["target"] for e in got] == ["new:latest"]


def test_log_action_respects_cap(tmp_path):
    p = tmp_path / "events.jsonl"
    clk = _clock()
    for i in range(8):
        audit.log_action(action="warm", target=str(i), kind="ollama",
                         now_fn=clk, path=p, cap=5)
    targets = [e["target"] for e in audit.read_events(limit=99, path=p)]
    assert targets == ["7:latest", "6:latest", "5:latest", "4:latest", "3:latest"]


def _holder(key, target, kind, size_mb=None):
    return {"key": key, "target": target, "kind": kind, "size_mb": size_mb}


def test_meaningful_holders_all_models_plus_big_processes():
    loaded = [{"name": "qwen3:8b"}]
    procs = [{"pid": 100, "size_mb": 14492, "name": "python.exe", "cmdline": "python train.py", "kind": "compute"},
             {"pid": 200, "size_mb": 40, "name": "chrome.exe", "cmdline": "chrome", "kind": "graphics"}]
    holders = audit.meaningful_holders(loaded, procs, threshold_mb=512)
    keys = {h["key"] for h in holders}
    assert "ollama:qwen3:8b" in keys
    assert "process:100" in keys        # 14.5 GB -> meaningful
    assert "process:200" not in keys    # 40 MB -> noise, filtered


@pytest.mark.parametrize("name, label", [("python.exe", "python.exe (pid 100)"),
                                         (None, "pid 100")])
def test_process_holder_label_never_carries_the_command_line(name, label):
    """The label is persisted and returned by history(); arguments can be secrets."""
    procs = [{"pid": 100, "size_mb": 14492, "name": name,
              "cmdline": "python serve.py --api-key=SECRET", "kind": "compute"}]
    [holder] = audit.meaningful_holders([], procs, threshold_mb=512)
    assert holder["target"] == label


def test_detect_first_run_sets_baseline_no_events(tmp_path):
    ls, log = tmp_path / "last_seen.json", tmp_path / "events.jsonl"
    holders = [_holder("ollama:qwen3:8b", "qwen3:8b", "ollama")]
    emitted = audit.detect_and_log(holders, last_seen_path=ls, log_path=log, now_fn=_clock())
    assert emitted == []                       # no false "everything disappeared"
    assert audit.read_events(path=log) == []


def test_detect_disappearance_of_process_is_unattributed(tmp_path):
    ls, log = tmp_path / "last_seen.json", tmp_path / "events.jsonl"
    clk = _clock()
    p = [_holder("process:100", "python.exe train_lora_kg.py", "process", 14492)]
    audit.detect_and_log(p, last_seen_path=ls, log_path=log, now_fn=clk)   # baseline
    clk.tick(30)
    emitted = audit.detect_and_log([], last_seen_path=ls, log_path=log, now_fn=clk)  # gone
    assert len(emitted) == 1
    e = emitted[0]
    assert e["type"] == "disappeared" and e["kind"] == "process"
    assert e["cause"] == "unattributed" and e["size_mb"] == 14492
    assert "python.exe train_lora_kg.py" in e["target"]


def test_detect_ollama_disappearance_attributed_to_recent_unload(tmp_path):
    ls, log = tmp_path / "last_seen.json", tmp_path / "events.jsonl"
    clk = _clock()
    m = [_holder("ollama:qwen3:8b", "qwen3:8b", "ollama")]
    audit.detect_and_log(m, last_seen_path=ls, log_path=log, now_fn=clk)   # baseline
    clk.tick(2)
    audit.log_action(action="unload", target="qwen3:8b", kind="ollama",
                     actor="retro-repo", now_fn=clk, path=log)             # our action
    clk.tick(1)
    emitted = audit.detect_and_log([], last_seen_path=ls, log_path=log, now_fn=clk)
    assert emitted[0]["cause"] == "self_action" and emitted[0]["actor"] == "retro-repo"


def test_detect_ollama_disappearance_without_action_is_external(tmp_path):
    ls, log = tmp_path / "last_seen.json", tmp_path / "events.jsonl"
    clk = _clock()
    m = [_holder("ollama:qwen3:8b", "qwen3:8b", "ollama")]
    audit.detect_and_log(m, last_seen_path=ls, log_path=log, now_fn=clk)
    clk.tick(30)
    emitted = audit.detect_and_log([], last_seen_path=ls, log_path=log, now_fn=clk)
    assert emitted[0]["cause"] == "external"


def test_detect_appearance_logged(tmp_path):
    ls, log = tmp_path / "last_seen.json", tmp_path / "events.jsonl"
    clk = _clock()
    audit.detect_and_log([], last_seen_path=ls, log_path=log, now_fn=clk)  # empty baseline
    m = [_holder("ollama:llama3.2", "llama3.2", "ollama")]
    emitted = audit.detect_and_log(m, last_seen_path=ls, log_path=log, now_fn=clk)
    assert emitted[0]["type"] == "appeared" and emitted[0]["target"] == "llama3.2:latest"


def test_detect_corrupt_baseline_treated_as_first_run_no_storm(tmp_path):
    ls, log = tmp_path / "last_seen.json", tmp_path / "events.jsonl"
    ls.write_text("garbage {", encoding="utf-8")
    clk = _clock()
    p = [_holder("process:100", "python train.py", "process", 14492)]
    emitted = audit.detect_and_log(p, last_seen_path=ls, log_path=log, now_fn=clk)
    assert emitted == []                       # corrupt baseline -> fresh first run, no storm
    assert audit.read_events(path=log) == []
    clk.tick(30)
    gone = audit.detect_and_log([], last_seen_path=ls, log_path=log, now_fn=clk)
    assert len(gone) == 1
    assert gone[0]["type"] == "disappeared" and gone[0]["cause"] == "unattributed"


def test_detect_valid_empty_baseline_still_fires_appeared(tmp_path):
    ls, log = tmp_path / "last_seen.json", tmp_path / "events.jsonl"
    ls.write_text('{"holders": {}}', encoding="utf-8")
    clk = _clock()
    m = [_holder("ollama:llama3.2", "llama3.2", "ollama")]
    emitted = audit.detect_and_log(m, last_seen_path=ls, log_path=log, now_fn=clk)
    assert len(emitted) == 1
    assert emitted[0]["type"] == "appeared" and emitted[0]["target"] == "llama3.2:latest"


def test_detect_second_session_does_not_double_log(tmp_path):
    ls, log = tmp_path / "last_seen.json", tmp_path / "events.jsonl"
    clk = _clock()
    m = [_holder("ollama:qwen3:8b", "qwen3:8b", "ollama")]
    audit.detect_and_log(m, last_seen_path=ls, log_path=log, now_fn=clk)
    clk.tick(30)
    audit.detect_and_log([], last_seen_path=ls, log_path=log, now_fn=clk)  # session A logs it
    again = audit.detect_and_log([], last_seen_path=ls, log_path=log, now_fn=clk)  # session B
    assert again == []  # already gone from last_seen -> nothing to re-log


def test_unsuccessful_actions_never_claim_a_disappearance(tmp_path):
    for outcome in ("refused", "failed", "unknown"):
        case = tmp_path / outcome
        ls, log = case / "last_seen.json", case / "events.jsonl"
        clk = _clock()
        holder = [_holder("ollama:llama3.2", "llama3.2", "ollama")]
        audit.detect_and_log(holder, last_seen_path=ls, log_path=log, now_fn=clk)
        clk.tick(2)
        audit.log_action(action="unload", target="llama3.2", kind="ollama",
                         outcome=outcome, now_fn=clk, path=log)
        clk.tick(1)
        [event] = audit.detect_and_log(
            [], last_seen_path=ls, log_path=log, now_fn=clk,
            observed_kinds={"ollama"},
        )
        assert event["cause"] == "external"


def test_unavailable_and_partial_reads_preserve_last_good_baseline(tmp_path):
    ls, log = tmp_path / "last_seen.json", tmp_path / "events.jsonl"
    clk = _clock()
    holders = [
        _holder("ollama:qwen3:8b", "qwen3:8b", "ollama"),
        _holder("process:100", "python train.py", "process", 12000),
    ]
    audit.detect_and_log(holders, last_seen_path=ls, log_path=log, now_fn=clk)

    clk.tick(10)
    assert audit.detect_and_log(
        None, last_seen_path=ls, log_path=log, now_fn=clk,
        observed_kinds={"ollama"},
    ) == []
    # Ollama succeeded with an empty result; process telemetry was unavailable.
    [ollama_gone] = audit.detect_and_log(
        [], last_seen_path=ls, log_path=log, now_fn=clk,
        observed_kinds={"ollama"},
    )
    assert ollama_gone["kind"] == "ollama"

    clk.tick(10)
    assert audit.detect_and_log(
        [holders[1]], last_seen_path=ls, log_path=log, now_fn=clk,
        observed_kinds={"process"},
    ) == []
    [process_gone] = audit.detect_and_log(
        [], last_seen_path=ls, log_path=log, now_fn=clk,
        observed_kinds={"process"},
    )
    assert process_gone["kind"] == "process"


def test_stale_source_timestamp_cannot_regress_baseline(tmp_path):
    ls, log = tmp_path / "last_seen.json", tmp_path / "events.jsonl"
    old = [_holder("ollama:old", "old", "ollama")]
    new = [_holder("ollama:new", "new", "ollama")]
    audit.detect_and_log(
        old, last_seen_path=ls, log_path=log, now_fn=lambda: _T0,
        observed_at=_T0, observed_kinds={"ollama"}, scope="ollama:local",
    )
    t20 = _T0 + timedelta(seconds=20)
    audit.detect_and_log(
        new, last_seen_path=ls, log_path=log, now_fn=lambda: t20,
        observed_at=t20, observed_kinds={"ollama"}, scope="ollama:local",
    )

    stale = audit.detect_and_log(
        [], last_seen_path=ls, log_path=log, now_fn=lambda: t20,
        observed_at=_T0 + timedelta(seconds=10), observed_kinds={"ollama"},
        scope="ollama:local",
    )
    assert stale == []
    assert audit.detect_and_log(
        new, last_seen_path=ls, log_path=log,
        now_fn=lambda: _T0 + timedelta(seconds=30),
        observed_at=_T0 + timedelta(seconds=30), observed_kinds={"ollama"},
        scope="ollama:local",
    ) == []


def test_scoped_baselines_do_not_cross_contaminate(tmp_path):
    ls, log = tmp_path / "last_seen.json", tmp_path / "events.jsonl"
    clk = _clock()
    holder = [_holder("ollama:qwen3:8b", "qwen3:8b", "ollama")]
    audit.detect_and_log(holder, last_seen_path=ls, log_path=log, now_fn=clk,
                         observed_kinds={"ollama"}, scope="ollama:a")
    audit.detect_and_log([], last_seen_path=ls, log_path=log, now_fn=clk,
                         observed_kinds={"ollama"}, scope="ollama:b")
    clk.tick(20)
    [gone] = audit.detect_and_log(
        [], last_seen_path=ls, log_path=log, now_fn=clk,
        observed_kinds={"ollama"}, scope="ollama:a",
    )
    assert gone["scope"] == "ollama:a"
    assert audit.detect_and_log(
        [], last_seen_path=ls, log_path=log, now_fn=clk,
        observed_kinds={"ollama"}, scope="ollama:b",
    ) == []


def test_action_from_another_scope_cannot_claim_disappearance(tmp_path):
    ls, log = tmp_path / "last_seen.json", tmp_path / "events.jsonl"
    clk = _clock()
    holder = [_holder("ollama:qwen3:8b", "qwen3:8b", "ollama")]
    audit.detect_and_log(holder, last_seen_path=ls, log_path=log, now_fn=clk,
                         observed_kinds={"ollama"}, scope="http://ollama-a")
    clk.tick(2)
    audit.log_action(action="unload", target="qwen3:8b", kind="ollama",
                     outcome="succeeded", scope="http://ollama-b",
                     now_fn=clk, path=log)
    clk.tick(1)
    [event] = audit.detect_and_log(
        [], last_seen_path=ls, log_path=log, now_fn=clk,
        observed_kinds={"ollama"}, scope="http://ollama-a",
    )
    assert event["cause"] == "external"


def test_legacy_ok_action_attributes_canonical_bare_name(tmp_path):
    ls, log = tmp_path / "last_seen.json", tmp_path / "events.jsonl"
    clk = _clock()
    holder = [_holder("ollama:llama3.2", "llama3.2", "ollama")]
    audit.detect_and_log(holder, last_seen_path=ls, log_path=log, now_fn=clk)
    clk.tick(2)
    log.write_text(
        '{"ts":"2026-07-16T18:00:02Z","type":"action","kind":"ollama",'
        '"target":"llama3.2","action":"unload","actor":"legacy","outcome":"ok"}\n',
        encoding="utf-8",
    )
    clk.tick(1)
    [event] = audit.detect_and_log([], last_seen_path=ls, log_path=log, now_fn=clk)
    assert event["target"] == "llama3.2:latest"
    assert event["cause"] == "self_action" and event["actor"] == "legacy"


def test_future_successful_action_is_not_used_for_attribution(tmp_path):
    ls, log = tmp_path / "last_seen.json", tmp_path / "events.jsonl"
    clk = _clock()
    holder = [_holder("ollama:qwen3:8b", "qwen3:8b", "ollama")]
    audit.detect_and_log(holder, last_seen_path=ls, log_path=log, now_fn=clk)
    future = _T0 + timedelta(seconds=30)
    audit.log_action(action="unload", target="qwen3:8b", kind="ollama",
                     outcome="succeeded", now_fn=lambda: future, path=log)
    clk.tick(1)
    [event] = audit.detect_and_log([], last_seen_path=ls, log_path=log, now_fn=clk)
    assert event["cause"] == "external"


def _sample(free_mb=1000, state="ok"):
    return {"free_mb": free_mb, "used_mb": 23000, "total_mb": 24576,
            "non_local_mb": 0, "state": state, "loaded_count": 1}


def test_first_sample_is_always_written(tmp_path):
    log, state = tmp_path / "e.jsonl", tmp_path / "s.json"
    now = datetime(2026, 7, 19, 12, 0, 0, tzinfo=timezone.utc)
    assert audit.maybe_log_sample(
        _sample(), state_path=state, log_path=log, now_fn=lambda: now) is True
    rows = audit.read_events(type="sample", path=log)
    assert len(rows) == 1
    assert rows[0]["free_mb"] == 1000


def test_sample_throttled_within_interval(tmp_path):
    log, state = tmp_path / "e.jsonl", tmp_path / "s.json"
    t0 = datetime(2026, 7, 19, 12, 0, 0, tzinfo=timezone.utc)
    audit.maybe_log_sample(_sample(), state_path=state, log_path=log,
                           now_fn=lambda: t0, interval_seconds=60)
    t1 = t0 + timedelta(seconds=30)
    assert audit.maybe_log_sample(
        _sample(), state_path=state, log_path=log,
        now_fn=lambda: t1, interval_seconds=60) is False
    assert len(audit.read_events(type="sample", path=log)) == 1


def test_sample_written_after_interval(tmp_path):
    log, state = tmp_path / "e.jsonl", tmp_path / "s.json"
    t0 = datetime(2026, 7, 19, 12, 0, 0, tzinfo=timezone.utc)
    audit.maybe_log_sample(_sample(), state_path=state, log_path=log,
                           now_fn=lambda: t0, interval_seconds=60)
    t1 = t0 + timedelta(seconds=61)
    assert audit.maybe_log_sample(
        _sample(free_mb=200), state_path=state, log_path=log,
        now_fn=lambda: t1, interval_seconds=60) is True
    assert len(audit.read_events(type="sample", path=log)) == 2


def test_sample_throttle_and_history_are_scoped(tmp_path):
    log, state = tmp_path / "e.jsonl", tmp_path / "s.json"
    now = datetime(2026, 7, 19, 12, 0, 0, tzinfo=timezone.utc)
    assert audit.maybe_log_sample(
        _sample(), state_path=state, log_path=log, now_fn=lambda: now,
        scope="gpu:0",
    ) is True
    assert audit.maybe_log_sample(
        _sample(), state_path=state, log_path=log, now_fn=lambda: now,
        scope="gpu:1",
    ) is True
    assert audit.maybe_log_sample(
        _sample(), state_path=state, log_path=log, now_fn=lambda: now,
        scope="gpu:0",
    ) is False
    assert len(audit.read_events(type="sample", scope="gpu:0", path=log)) == 1
    assert len(audit.read_events(type="sample", scope="gpu:1", path=log)) == 1


def test_corrupt_state_file_treated_as_never_sampled(tmp_path):
    log, state = tmp_path / "e.jsonl", tmp_path / "s.json"
    state.write_text("{not json", encoding="utf-8")
    now = datetime(2026, 7, 19, 12, 0, 0, tzinfo=timezone.utc)
    assert audit.maybe_log_sample(
        _sample(), state_path=state, log_path=log, now_fn=lambda: now) is True


def _boom(*_a, **_k):
    raise OSError("disk on fire")


def test_maybe_log_sample_never_raises_on_state_write_failure(tmp_path, monkeypatch):
    # A failing state write must degrade to "no sample", never to an exception:
    # the audit may never break a tool call.
    monkeypatch.setattr(audit, "save_json_atomic", _boom)
    assert audit.maybe_log_sample(
        _sample(), state_path=tmp_path / "s.json",
        log_path=tmp_path / "e.jsonl") is False


def test_maybe_log_sample_never_raises_on_event_append_failure(tmp_path, monkeypatch):
    # Same contract on the far side of the throttle: the events append is the
    # second, independently-locked write and it can fail on its own.
    monkeypatch.setattr(audit, "append_jsonl_capped", _boom)
    assert audit.maybe_log_sample(
        _sample(), state_path=tmp_path / "s.json",
        log_path=tmp_path / "e.jsonl") is False


def test_maybe_log_sample_never_raises_on_a_non_mapping_sample(tmp_path):
    # ``**sample`` is the one unguarded-looking expression in the function; a
    # caller passing garbage must still not break the tool call.
    assert audit.maybe_log_sample(
        None, state_path=tmp_path / "s.json",
        log_path=tmp_path / "e.jsonl") is False


def test_maybe_log_sample_survives_a_directory_where_the_state_file_belongs(tmp_path):
    # Windows os.replace happily renames a directory when the destination is
    # free, so load_json quarantines it aside and the write actually succeeds.
    # The contract under test is only that we return a bool instead of raising.
    state = tmp_path / "s.json"
    state.mkdir()
    result = audit.maybe_log_sample(
        _sample(), state_path=state, log_path=tmp_path / "e.jsonl")
    assert isinstance(result, bool)


def test_summarize_samples_reports_falling_trend():
    rows = [{"free_mb": v, "state": "ok"} for v in (9000, 8000, 3000, 1000)]
    s = audit.summarize_samples(rows)
    assert s["direction"] == "falling"
    assert s["min_free_mb"] == 1000
    assert s["max_free_mb"] == 9000
    assert s["latest_free_mb"] == 1000
    assert s["count"] == 4


def test_summarize_samples_counts_spilling():
    rows = [{"free_mb": 100, "state": "thrashing"},
            {"free_mb": 100, "state": "ok"},
            {"free_mb": 100, "state": "thrashing"}]
    assert audit.summarize_samples(rows)["thrashing_samples"] == 2


def test_summarize_samples_latest_is_none_when_newest_row_has_no_value():
    """"now N MB" must describe the NEWEST row, not the newest row that
    happened to carry a number — otherwise the summary reports a stale value
    as the current one."""
    rows = [{"free_mb": 9000, "state": "ok"},
            {"free_mb": 1000, "state": "ok"},
            {"free_mb": None, "state": "ok"}]     # newest row, unmeasured
    s = audit.summarize_samples(rows)
    assert s["latest_free_mb"] is None
    assert s["min_free_mb"] == 1000              # min/max still use real values
    assert s["max_free_mb"] == 9000
    assert s["count"] == 3


def test_summarize_samples_handles_empty_and_none():
    assert audit.summarize_samples([])["count"] == 0
    assert audit.summarize_samples([{"free_mb": None}])["direction"] == "unknown"
