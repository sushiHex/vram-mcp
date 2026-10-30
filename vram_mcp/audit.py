"""Append-only, cause-attributed VRAM audit log + disappearance detector.

One typed ``events.jsonl`` plus scoped holder baselines in ``last_seen.json``.
Bounded (``cap`` events, pruned on append), crash-safe (atomic writes + the
shared file lock). The audit MUST NEVER break a tool call: every write is
best-effort. Pure w.r.t. time (``now_fn`` injected) and paths (injected in
tests).
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

from ._util import STATE_DIR, append_jsonl_capped, iso, load_json, locked, parse_iso, read_jsonl, save_json_atomic
from .models import canonical_model

DEFAULT_EVENTS_PATH = STATE_DIR / "events.jsonl"
DEFAULT_LAST_SEEN_PATH = STATE_DIR / "last_seen.json"
DEFAULT_SAMPLE_PATH = STATE_DIR / "last_sample.json"
SAMPLE_INTERVAL_SECONDS = 60.0

_RECENT_ACTION_SECONDS = 10.0
_HOLDER_KINDS = frozenset({"ollama", "process"})
_ACTION_OUTCOMES = frozenset({"succeeded", "refused", "failed", "unknown"})


def _default_now() -> datetime:
    return datetime.now(timezone.utc)


def _canonical_target(target, kind: str):
    if kind != "ollama":
        return target
    try:
        return canonical_model(target)
    except (TypeError, ValueError):
        return target


def _normalized_outcome(outcome: str) -> str:
    # ``ok`` was written by releases before action outcomes were standardized.
    if outcome == "ok":
        return "succeeded"
    return outcome if outcome in _ACTION_OUTCOMES else "unknown"


_CORE_KEYS = frozenset({"ts", "type", "kind", "target", "action", "actor", "force",
                        "outcome", "detail", "scope"})
_TOKEN = re.compile(r"[A-Za-z0-9_.:+-]{1,64}")


def _is_audit_value(value) -> bool:
    if value is None or isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, str):
        return _TOKEN.fullmatch(value) is not None
    if isinstance(value, list):
        return all(isinstance(v, str) and _TOKEN.fullmatch(v) for v in value)
    return False


def log_action(*, action: str, target: str, kind: str, actor: str = "unknown",
               force: bool = False, outcome: str = "succeeded", detail: str = "",
               scope: str | None = None, fields: dict | None = None,
               now_fn=_default_now, path: Path = DEFAULT_EVENTS_PATH,
               cap: int = 5000) -> None:
    """Append one ``type="action"`` event. Best-effort — never raises.

    ``fields`` adds structured readings: numbers, flags, and short tokens
    (categories, class names, timestamps) or lists of them. Anything else is
    dropped, as is any field named like a core key, so a producer cannot write
    free text, a path or a prompt into the log through them.
    """
    try:
        event = {key: value for key, value in (fields or {}).items()
                 if key not in _CORE_KEYS and _is_audit_value(value)}
        event.update({"ts": iso(now_fn()), "type": "action", "kind": kind,
                      "target": _canonical_target(target, kind),
                      "action": action, "actor": actor, "force": force,
                      "outcome": _normalized_outcome(outcome), "detail": detail})
        if scope is not None:
            event["scope"] = scope
        with locked(path):
            append_jsonl_capped(path, event, cap)
    except Exception:
        pass  # the audit may never break a tool call


def read_events(*, model: str | None = None, type: str | None = None,
                limit: int = 50, since: str | None = None,
                scope: str | None = None,
                path: Path = DEFAULT_EVENTS_PATH) -> list[dict]:
    """Recent events, newest-first, optionally filtered by their fields."""
    rows = read_jsonl(path)
    if model is not None:
        canonical = _canonical_target(model, "ollama")
        rows = [r for r in rows
                if (_canonical_target(r.get("target"), r.get("kind")) == canonical
                    if r.get("kind") == "ollama" else r.get("target") == model)]
    if type is not None:
        rows = [r for r in rows if r.get("type") == type]
    if since is not None:
        rows = [r for r in rows if r.get("ts", "") >= since]  # ISO-Z sorts lexically
    if scope is not None:
        rows = [r for r in rows if r.get("scope") == scope]
    rows.reverse()
    return rows[:limit]


def meaningful_holders(loaded_models, process_table, threshold_mb: int) -> list[dict]:
    """Holders worth tracking: every Ollama model, plus non-Ollama processes
    holding >= ``threshold_mb`` dedicated VRAM (size is the filter, not names)."""
    holders = []
    for m in loaded_models:
        name = m.get("name")
        if not name:
            continue
        name = _canonical_target(name, "ollama")
        holders.append({"key": f"ollama:{name}", "target": name,
                        "kind": "ollama", "size_mb": m.get("size_vram_mb")})
    for p in process_table:
        size = p.get("size_mb")
        if size is not None and size >= threshold_mb:
            # Never the command line: it can carry credentials, and this label
            # is persisted and returned by history().
            label = f"{p['name']} (pid {p['pid']})" if p.get("name") else f"pid {p['pid']}"
            holders.append({"key": f"process:{p['pid']}", "target": label,
                            "kind": "process", "size_mb": size})
    return holders


def _recent_action_for(target: str, now: datetime, log_path: Path,
                       scope: str | None = None):
    """The most recent unload/ensure_free action on ``target`` within the
    attribution window, else None. ``read_events`` is newest-first."""
    floor = now.timestamp() - _RECENT_ACTION_SECONDS
    for e in read_events(model=target, type="action", limit=5000, path=log_path):
        # An action against another Ollama endpoint cannot explain this scope.
        # Legacy/default observations only match legacy/default actions.
        if e.get("scope") != scope:
            continue
        if e.get("action") not in ("unload", "ensure_free"):
            continue
        if e.get("outcome") not in ("succeeded", "ok"):
            continue
        ts = e.get("ts")
        if not ts:
            continue
        try:
            when = parse_iso(ts).timestamp()
        except ValueError:
            continue
        if floor <= when <= now.timestamp():
            return e
    return None


def _disappeared_event(holder: dict, now: datetime, log_path: Path,
                       scope: str | None = None) -> dict:
    kind = holder["kind"]
    if kind == "ollama":
        action = _recent_action_for(holder["target"], now, log_path, scope)
        if action is not None:
            event = {"ts": iso(now), "type": "disappeared", "kind": "ollama",
                    "target": holder["target"], "size_mb": holder.get("size_mb"),
                    "cause": "self_action", "actor": action.get("actor", "unknown"),
                    "detail": f"removed by {action.get('actor','unknown')} via {action.get('action')}"}
            if scope is not None:
                event["scope"] = scope
            return event
        event = {"ts": iso(now), "type": "disappeared", "kind": "ollama",
                "target": holder["target"], "size_mb": holder.get("size_mb"),
                "cause": "external",
                "detail": "no vram-mcp action recorded — Ollama idle-expiry, "
                          "memory-pressure eviction, or an external unload."}
        if scope is not None:
            event["scope"] = scope
        return event
    event = {"ts": iso(now), "type": "disappeared", "kind": "process",
            "target": holder["target"], "size_mb": holder.get("size_mb"),
            "cause": "unattributed",
            "detail": "process exited or was killed; vram-mcp cannot observe the cause."}
    if scope is not None:
        event["scope"] = scope
    return event


def _valid_baseline_exists(path: Path) -> bool:
    """True only if ``path`` holds a readable, correctly-shaped baseline.
    Missing, unreadable, or wrong-shape all count as 'no baseline' -> a fresh
    first run that must NOT emit appeared/disappeared events."""
    if not path.exists():
        return False
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(doc, dict) and isinstance(doc.get("holders"), dict)


def _normalize_holder(holder: dict) -> dict:
    normalized = dict(holder)
    kind = normalized.get("kind")
    target = _canonical_target(normalized.get("target"), kind)
    normalized["target"] = target
    if kind == "ollama":
        normalized["key"] = f"ollama:{target}"
    return normalized


def _normalize_holder_map(holders: dict) -> dict:
    normalized = {}
    for holder in holders.values():
        if not isinstance(holder, dict) or holder.get("kind") not in _HOLDER_KINDS:
            continue
        item = _normalize_holder(holder)
        key = item.get("key")
        if key:
            normalized[key] = item
    return normalized


def _observation_time(value, fallback: datetime) -> datetime:
    when = fallback if value is None else (parse_iso(value) if isinstance(value, str) else value)
    if not isinstance(when, datetime):
        raise TypeError("observed_at must be a datetime or ISO timestamp")
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.astimezone(timezone.utc)


def _observation_iso(when: datetime) -> str:
    """Preserve source precision so near-concurrent reads order correctly."""
    return when.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _baseline_domain(state: dict, scope: str | None,
                     default_existed: bool) -> tuple[dict, bool]:
    """Return a baseline domain and whether it existed before this read."""
    if scope is None:
        return state, default_existed
    scopes = state.setdefault("scopes", {})
    if not isinstance(scopes, dict):
        scopes = {}
        state["scopes"] = scopes
    existing = scopes.get(scope)
    if isinstance(existing, dict) and isinstance(existing.get("holders"), dict):
        return existing, True
    domain = {"holders": {}, "observed_at": {}, "initialized_kinds": []}
    scopes[scope] = domain
    return domain, False


def detect_and_log(current_holders, *, last_seen_path: Path = DEFAULT_LAST_SEEN_PATH,
                   log_path: Path = DEFAULT_EVENTS_PATH, now_fn=_default_now,
                   cap: int = 5000, observed_at=None,
                   observed_kinds: set[str] | None = None,
                   scope: str | None = None) -> list[dict]:
    """Diff one successful observation against its scoped last-seen baseline.

    ``None`` means the source was unavailable; a successful empty result is
    represented by ``[]``. Only ``observed_kinds`` are diffed and replaced.
    Older source timestamps are ignored. Returns emitted events and never
    raises out to the caller.
    """
    emitted: list[dict] = []
    try:
        # ``None`` is the unavailable-reading sentinel. It must not turn every
        # last-known holder into a disappearance.
        if current_holders is None:
            return []
        kinds = (_HOLDER_KINDS if observed_kinds is None
                 else frozenset(observed_kinds) & _HOLDER_KINDS)
        if not kinds:
            return []
        with locked(last_seen_path):
            now = now_fn()
            source_time = _observation_time(observed_at, now)
            state_existed = _valid_baseline_exists(last_seen_path)
            prev = load_json(last_seen_path, lambda: {"holders": {}},
                             lambda d: isinstance(d, dict) and isinstance(d.get("holders"), dict))
            domain, domain_existed = _baseline_domain(prev, scope, state_existed)
            prev_map = _normalize_holder_map(domain["holders"])
            cur_map = {}
            for holder in current_holders:
                if not isinstance(holder, dict) or holder.get("kind") not in kinds:
                    continue
                normalized = _normalize_holder(holder)
                key = normalized.get("key")
                if key:
                    cur_map[key] = normalized

            recorded_times = domain.get("observed_at", {})
            if not isinstance(recorded_times, dict):
                recorded_times = {}
            initialized = domain.get("initialized_kinds")
            if not isinstance(initialized, list):
                # A legacy valid baseline represented successful observation of
                # both kinds, even though it did not save that metadata.
                initialized = list(_HOLDER_KINDS) if domain_existed else []
            initialized_set = set(initialized) & _HOLDER_KINDS

            accepted_kinds = set()
            for kind in kinds:
                previous_time = recorded_times.get(kind)
                if previous_time:
                    try:
                        if source_time < _observation_time(previous_time, now):
                            continue
                    except (TypeError, ValueError):
                        pass
                accepted_kinds.add(kind)

            if not accepted_kinds:
                return []

            for kind in sorted(accepted_kinds):
                old_for_kind = {key: h for key, h in prev_map.items()
                                if h.get("kind") == kind}
                new_for_kind = {key: h for key, h in cur_map.items()
                                if h.get("kind") == kind}
                for key, holder in old_for_kind.items():
                    if key not in new_for_kind:
                        emitted.append(_disappeared_event(holder, now, log_path, scope))
                if kind in initialized_set:
                    for key, holder in new_for_kind.items():
                        if key not in old_for_kind:
                            event = {
                                "ts": iso(now), "type": "appeared", "kind": holder["kind"],
                                "target": holder["target"], "size_mb": holder.get("size_mb"),
                                "detail": "now holding VRAM",
                            }
                            if scope is not None:
                                event["scope"] = scope
                            emitted.append(event)

                prev_map = {key: h for key, h in prev_map.items()
                            if h.get("kind") != kind}
                prev_map.update(new_for_kind)
                recorded_times[kind] = _observation_iso(source_time)
                initialized_set.add(kind)

            # Baseline is saved BEFORE we release the last_seen lock, so a
            # concurrent session already sees the holder gone and won't re-log it.
            domain["holders"] = prev_map
            domain["observed_at"] = recorded_times
            domain["initialized_kinds"] = sorted(initialized_set)
            save_json_atomic(last_seen_path, prev)
    except Exception:
        return emitted
    # Append events OUTSIDE the last_seen lock, under the EVENTS lock, so both
    # writers of events.jsonl (this function and log_action) are serialized.
    # Lock order is acyclic (last_seen is already released here; log_action
    # never takes the last_seen lock), and best-effort — a failed append never
    # breaks the caller.
    if emitted:
        try:
            with locked(log_path):
                for e in emitted:
                    append_jsonl_capped(log_path, e, cap)
        except Exception:
            pass
    return emitted


def maybe_log_sample(sample: dict, *, interval_seconds: float = SAMPLE_INTERVAL_SECONDS,
                     state_path: Path = DEFAULT_SAMPLE_PATH,
                     log_path: Path = DEFAULT_EVENTS_PATH,
                     now_fn=_default_now, cap: int = 5000,
                     scope: str | None = None) -> bool:
    """Append one ``type="sample"`` event, at most once per ``interval_seconds``
    for each scope across all sessions.

    The throttle matters: ``events.jsonl`` is capped, and an unthrottled sample
    on every status call would evict the action/disappearance events that carry
    the real diagnostic value. The last-sample timestamp lives in its own small
    state file under the shared lock, so concurrent sessions agree on the rate
    independently for each GPU scope.

    Returns True if a sample was written. Best-effort — never raises.
    """
    try:
        if not isinstance(sample, dict):
            return False
        effective_scope = scope if scope is not None else sample.get("scope")
        now = now_fn()
        with locked(state_path):
            state = load_json(state_path, lambda: {},
                              lambda d: isinstance(d, dict))
            if effective_scope is None:
                sample_state = state
            else:
                scopes = state.setdefault("scopes", {})
                if not isinstance(scopes, dict):
                    scopes = {}
                    state["scopes"] = scopes
                sample_state = scopes.setdefault(effective_scope, {})
                if not isinstance(sample_state, dict):
                    sample_state = {}
                    scopes[effective_scope] = sample_state
            last = sample_state.get("ts")
            if last:
                try:
                    elapsed = (now - parse_iso(last)).total_seconds()
                    if elapsed < interval_seconds:
                        return False
                except (ValueError, TypeError):
                    pass  # unparsable timestamp -> treat as never sampled
            sample_state["ts"] = iso(now)
            save_json_atomic(state_path, state)
    except Exception:
        return False
    # Appended OUTSIDE the state lock, under the EVENTS lock, so every writer of
    # events.jsonl is serialized by one lock and none are nested. Building the
    # event is inside the try too: ``**sample`` raises on a non-mapping, and
    # nothing here may escape to the caller.
    try:
        event = {**sample, "ts": iso(now), "type": "sample"}
        if effective_scope is not None:
            event["scope"] = effective_scope
        with locked(log_path):
            append_jsonl_capped(log_path, event, cap)
    except Exception:
        return False
    return True


def summarize_samples(rows: list[dict]) -> dict:
    """Reduce ``type="sample"`` events to a trend. Pure — no IO.

    ``direction`` compares the mean of the first third against the last third,
    which is robust to a single spike in a way that first-vs-last is not. Rows
    are expected oldest-first.

    ``latest_free_mb`` describes the NEWEST row, not the newest row that
    happened to carry a number: callers render it as "now N MB", and a stale
    value presented as the current one is a lie, where ``None`` ("unknown") is
    merely a gap.
    """
    values = [r.get("free_mb") for r in rows
              if isinstance(r, dict) and isinstance(r.get("free_mb"), int)]
    thrashing = sum(1 for r in rows
                    if isinstance(r, dict) and r.get("state") == "thrashing")
    newest = rows[-1] if rows else None
    latest = (newest.get("free_mb")
              if isinstance(newest, dict) and isinstance(newest.get("free_mb"), int)
              else None)
    if not values:
        return {"count": len(rows), "direction": "unknown",
                "min_free_mb": None, "max_free_mb": None,
                "latest_free_mb": None, "thrashing_samples": thrashing}

    third = max(1, len(values) // 3)
    head = sum(values[:third]) / third
    tail = sum(values[-third:]) / third
    delta = tail - head
    # 10% of the starting level, floored at 128 MB, so ordinary jitter on a
    # quiet GPU doesn't read as a trend.
    threshold = max(128.0, abs(head) * 0.10)
    if delta <= -threshold:
        direction = "falling"
    elif delta >= threshold:
        direction = "rising"
    else:
        direction = "flat"

    return {
        "count": len(rows),
        "direction": direction,
        "min_free_mb": min(values),
        "max_free_mb": max(values),
        "latest_free_mb": latest,
        "thrashing_samples": thrashing,
    }
