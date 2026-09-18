"""Durable, cross-session claims, reservations, and pending operations.

The JSON ledger is guarded only while it is read or changed. Long-running
backend work uses a durable operation lease instead of holding a file lock
across HTTP. A crashed caller can delay a same-model operation only until its
lease expires; it can never leave the ledger permanently locked.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

from ._util import (
    locked as _locked, iso as _iso, parse_iso as _parse_iso,
    save_json_atomic, _try_lock, _unlock,
)
from .models import canonical_model
from .validation import nonblank_text, positive_gb, positive_ttl_seconds

_DEFAULT_PATH = Path.home() / ".cache" / "vram-mcp" / "claims.json"
_OPERATION_LEASE_SECONDS = 120
_EVICTION_KINDS = {"unload", "ensure_free"}
_DEFAULT_SCOPE = "gpu:index=0"
_OPERATION_LOCKS: dict[str, int] = {}


def _default_now() -> datetime:
    return datetime.now(timezone.utc)


def _expires_at(now: datetime, ttl_seconds: int) -> datetime:
    try:
        return now + timedelta(seconds=ttl_seconds)
    except OverflowError as exc:
        raise ValueError("ttl_seconds exceeds the supported datetime range") from exc


def _operation_scope(record: dict) -> str:
    """Read legacy no-scope operation records as the original default GPU."""
    return nonblank_text(record.get("scope", _DEFAULT_SCOPE), "scope")


_LIFECYCLES = ("in_flight", "unknown")
_OUTCOMES = ("succeeded", "refused", "failed", "unknown")


def _identifiable_operation(record: object) -> bool:
    """Can this record be HONOURED — do we know what it protects, until when?

    Deliberately narrower than "do we understand it". A record written by a
    newer version may use a lifecycle or outcome this one has never heard of,
    which makes it uninterpretable, not ignorable: it still names a model, a
    scope and a lease. Dropping it would delete another process's in-flight
    protection and let this process evict straight past it — the one thing a
    ledger shared across versions must never do.

    Only a record whose identity or lease cannot be read at all is discarded,
    because there is then nothing left to honour.
    """
    try:
        if not isinstance(record, dict):
            raise ValueError
        nonblank_text(record.get("operation_id"), "operation_id")
        model = record.get("model")
        if not isinstance(model, str):
            raise ValueError
        canonical_model(model)
        nonblank_text(record.get("kind"), "kind")
        _operation_scope(record)
        _parse_iso(record["started_at"])
        expires_at = _parse_iso(record["expires_at"])
        if expires_at.tzinfo is None:
            raise ValueError
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _interpretable_operation(record: dict) -> bool:
    """Does THIS version understand the record's lifecycle vocabulary?

    False means a newer writer used terms we do not have. The record is still
    honoured; we decline to describe its lifecycle, outcome and retry state
    rather than guess at them — reporting a foreign state as ``in_flight``
    would be a claim about work we cannot actually see.
    """
    if record.get("lifecycle") not in (None, *_LIFECYCLES):
        return False
    if record.get("outcome") not in (None, *_OUTCOMES):
        return False
    retry_count = record.get("retry_count", 0)
    if isinstance(retry_count, bool) or not isinstance(retry_count, int) or retry_count < 0:
        return False
    return all(isinstance(record.get(key), (str, type(None)))
               for key in ("pending_until", "reason", "retry_after"))


def _load(path: Path) -> dict:
    """Read the ledger, retaining claims when an operation record is bad.

    Operations survive on identity alone, so a record this version cannot
    interpret round-trips intact instead of being erased by the next write.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {"claims": [], "operations": []}
    except OSError:
        raise
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise ValueError(f"claim ledger is corrupt: {path}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("claims"), list):
        raise ValueError(f"claim ledger has invalid shape: {path}")
    operations = data.get("operations", [])
    if not isinstance(operations, list):
        raise ValueError(f"claim ledger has invalid operations: {path}")
    data["operations"] = [r for r in operations if _identifiable_operation(r)]
    return data


def _save(path: Path, data: dict) -> None:
    save_json_atomic(path, data)


def _is_active(record: dict, now: datetime) -> bool:
    try:
        return isinstance(record, dict) and now < _parse_iso(record["expires_at"])
    except (KeyError, ValueError, TypeError):
        return False


def _operation_lock_path(path: Path, model: str) -> Path:
    digest = hashlib.sha256(model.encode("utf-8")).hexdigest()
    return path.with_suffix(path.suffix + f".operation-{digest}.lock")


def _try_operation_lock(path: Path, model: str) -> int | None:
    lock_path = _operation_lock_path(path, model)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    if os.fstat(fd).st_size == 0:
        os.write(fd, b"\0")
    try:
        _try_lock(fd)
    except BlockingIOError:
        os.close(fd)
        return None
    return fd


def _release_operation_lock(fd: int) -> None:
    try:
        _unlock(fd)
    finally:
        os.close(fd)


def _operation_is_live(path: Path, record: dict) -> bool:
    operation_id = record.get("operation_id") if isinstance(record, dict) else None
    if operation_id in _OPERATION_LOCKS:
        return True
    try:
        model = canonical_model(record.get("model"))
    except (AttributeError, ValueError):
        return False
    fd = _try_operation_lock(path, model)
    if fd is None:
        return True
    _release_operation_lock(fd)
    return False


def _operation_view(path: Path, record: dict, now: datetime) -> dict:
    """Project a ledger operation into the stable agent-facing status shape.

    ``lifecycle`` is ``"unrecognized"`` for a record this version cannot
    interpret. Its identity, scope and lease are still reported — that is what
    a caller needs in order to wait — while the interpretive fields stay null
    rather than being coerced into a state we did not observe.
    """
    expires_at = _parse_iso(record["expires_at"])
    owner_live = _operation_is_live(path, record)
    if not _interpretable_operation(record):
        lifecycle = "unrecognized"
    elif record.get("lifecycle") in _LIFECYCLES:
        lifecycle = record["lifecycle"]
    else:
        lifecycle = "in_flight"
    pending_until = record.get("pending_until")
    if not isinstance(pending_until, str) or not pending_until.strip():
        pending_until = record["expires_at"]
    outcome = "unknown" if lifecycle == "unknown" else None
    retry_count = record.get("retry_count", 0)
    if isinstance(retry_count, bool) or not isinstance(retry_count, int) or retry_count < 0:
        retry_count = 0
    retry_after = pending_until if lifecycle == "unknown" else None
    reason = record.get("reason")
    if not isinstance(reason, str):
        # A foreign record may carry any shape here; the schema promises a
        # string or null, and guessing at a translation would be worse.
        reason = None
    return {
        "operation_id": record["operation_id"],
        "model": canonical_model(record["model"]),
        "kind": record["kind"],
        "scope": _operation_scope(record),
        "started_at": record["started_at"],
        "expires_at": record["expires_at"],
        "pending_until": pending_until,
        "lifecycle": lifecycle,
        "owner_live": owner_live,
        "lease_expired": now >= expires_at,
        "outcome": outcome,
        "reason": reason,
        "retry_count": retry_count,
        "retry_after": retry_after,
    }


def _prune_expired(data: dict, now: datetime, path: Path) -> None:
    data["claims"] = [r for r in data["claims"] if _is_active(r, now)]
    # An elapsed wall-clock lease does not evict a still-live operation owner.
    # The per-operation OS lock proves liveness across processes; an absent lock
    # means a crashed holder's bounded lease is safe to discard.
    expired = []
    retained = []
    for record in data["operations"]:
        if _is_active(record, now) or _operation_is_live(path, record):
            retained.append(record)
        else:
            expired.append(record)
    data["operations"] = retained
    retry_counts = data.setdefault("operation_retry_counts", {})
    if not isinstance(retry_counts, dict):
        retry_counts = {}
        data["operation_retry_counts"] = retry_counts
    for record in expired:
        try:
            model = canonical_model(record["model"])
        except (KeyError, TypeError, ValueError):
            continue
        count = record.get("retry_count", 0)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            count = 0
        previous = retry_counts.get(model, 0)
        if isinstance(previous, bool) or not isinstance(previous, int) or previous < 0:
            previous = 0
        retry_counts[model] = max(previous, count + 1)


def _canonical_record(record: dict) -> dict | None:
    if not isinstance(record, dict):
        return None
    if record.get("kind") == "reservation":
        return dict(record)
    try:
        model = canonical_model(record.get("model"))
    except ValueError:
        return None
    return {**record, "model": model}


def _claim_view(record: dict) -> dict | None:
    """Return a claim entry, preserving protection when fields are malformed."""
    canonical = _canonical_record(record)
    if canonical is None:
        return None
    kind = "reservation" if canonical.get("kind") == "reservation" else "model"
    claim_id = canonical.get("claim_id")
    if not isinstance(claim_id, str) or not claim_id.strip():
        claim_id = None
    model = canonical.get("model") if kind == "model" else None
    text_fields = {
        key: value if isinstance(value, str) else None
        for key, value in ((key, canonical.get(key)) for key in (
            "owner", "purpose", "claimed_at", "renewed_at", "expires_at"))
    }
    ttl_seconds = canonical.get("ttl_seconds")
    if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int):
        ttl_seconds = None
    gb = canonical.get("gb")
    if kind != "reservation" or isinstance(gb, bool) or not isinstance(gb, (int, float)):
        gb = None
    else:
        try:
            gb = float(gb) if math.isfinite(float(gb)) else None
        except (OverflowError, TypeError, ValueError):
            gb = None
    pid = canonical.get("pid")
    if pid is not None and (isinstance(pid, bool) or not isinstance(pid, int)):
        pid = None
    return {
        "claim_id": claim_id,
        "kind": kind,
        "model": model,
        "pid": pid,
        "gb": gb,
        **text_fields,
        "ttl_seconds": ttl_seconds,
    }


def _active_model_claims(data: dict, model: str, now: datetime) -> list[dict]:
    result = []
    for record in data["claims"]:
        if not _is_active(record, now):
            continue
        canonical = _canonical_record(record)
        if canonical is not None and canonical.get("model") == model:
            result.append(canonical)
    return result


def _pending_for(data: dict, model: str, now: datetime) -> list[dict]:
    result = []
    for record in data["operations"]:
        try:
            same_model = canonical_model(record.get("model")) == model
        except (AttributeError, ValueError):
            same_model = False
        if same_model:
            result.append(dict(record))
    return result


def _pending_warm_for_scope(data: dict, scope: str) -> list[dict]:
    """Warm operations reserve admission capacity for their whole GPU scope."""
    return [dict(record) for record in data["operations"]
            if record.get("kind") == "warm" and _operation_scope(record) == scope]


def _has_pending_warm(data: dict) -> bool:
    return any(record.get("kind") == "warm" for record in data["operations"])


def claim(
    model: str, owner: str, purpose: str, ttl_seconds: int = 3600, *,
    path: Optional[Path] = None, now_fn: Callable[[], datetime] = _default_now,
) -> dict:
    """Register a model claim unless an operation has already begun on it."""
    path = path or _DEFAULT_PATH
    model = canonical_model(model)
    owner = nonblank_text(owner, "owner")
    purpose = nonblank_text(purpose, "purpose")
    ttl_seconds = positive_ttl_seconds(ttl_seconds)
    now = now_fn()
    expires_at = _expires_at(now, ttl_seconds)
    record = {
        "claim_id": uuid.uuid4().hex, "kind": "model", "model": model,
        "owner": owner, "purpose": purpose, "claimed_at": _iso(now),
        "renewed_at": _iso(now), "ttl_seconds": ttl_seconds,
        "expires_at": _iso(expires_at),
    }
    with _locked(path):
        data = _load(path)
        _prune_expired(data, now, path)
        if _pending_for(data, model, now):
            raise ValueError(f"operation pending for {model}")
        data["claims"].append(record)
        _save(path, data)
    return {"claim_id": record["claim_id"], "expires_at": record["expires_at"],
            "model": model}


def reserve(
    gb: float, owner: str, purpose: str, ttl_seconds: int = 3600, *,
    pid: Optional[int] = None, path: Optional[Path] = None,
    now_fn: Callable[[], datetime] = _default_now,
) -> dict:
    """Reserve finite positive VRAM capacity for a finite positive TTL."""
    path = path or _DEFAULT_PATH
    gb = positive_gb(gb)
    owner = nonblank_text(owner, "owner")
    purpose = nonblank_text(purpose, "purpose")
    ttl_seconds = positive_ttl_seconds(ttl_seconds)
    now = now_fn()
    expires_at = _expires_at(now, ttl_seconds)
    record = {
        "claim_id": uuid.uuid4().hex, "kind": "reservation", "model": None,
        "gb": gb, "pid": pid, "owner": owner, "purpose": purpose,
        "claimed_at": _iso(now), "renewed_at": _iso(now),
        "ttl_seconds": ttl_seconds, "expires_at": _iso(expires_at),
    }
    with _locked(path):
        data = _load(path)
        _prune_expired(data, now, path)
        if _has_pending_warm(data):
            raise ValueError("warm admission pending; retry reservation after it completes")
        data["claims"].append(record)
        _save(path, data)
    return {"claim_id": record["claim_id"], "expires_at": record["expires_at"]}


def renew(
    claim_id: str, ttl_seconds: Optional[int] = None, *,
    path: Optional[Path] = None, now_fn: Callable[[], datetime] = _default_now,
) -> dict:
    """Extend a live claim; malformed legacy TTLs are rejected, never guessed."""
    path = path or _DEFAULT_PATH
    claim_id = nonblank_text(claim_id, "claim_id")
    requested_ttl = positive_ttl_seconds(ttl_seconds) if ttl_seconds is not None else None
    now = now_fn()
    with _locked(path):
        data = _load(path)
        for record in data["claims"]:
            if isinstance(record, dict) and record.get("claim_id") == claim_id:
                if not _is_active(record, now):
                    return {"ok": False}
                ttl = requested_ttl if requested_ttl is not None else positive_ttl_seconds(
                    record.get("ttl_seconds"))
                record["ttl_seconds"] = ttl
                record["renewed_at"] = _iso(now)
                record["expires_at"] = _iso(_expires_at(now, ttl))
                _prune_expired(data, now, path)
                _save(path, data)
                return {"ok": True, "expires_at": record["expires_at"]}
    return {"ok": False}


def release(
    claim_id: str, *, path: Optional[Path] = None,
    now_fn: Callable[[], datetime] = _default_now,
) -> dict:
    """Remove a named claim and prune expired records under the same lock."""
    path = path or _DEFAULT_PATH
    claim_id = nonblank_text(claim_id, "claim_id")
    now = now_fn()
    with _locked(path):
        data = _load(path)
        before = len(data["claims"])
        data["claims"] = [
            r for r in data["claims"]
            if not (isinstance(r, dict) and r.get("claim_id") == claim_id)
        ]
        found = len(data["claims"]) != before
        if found:
            _prune_expired(data, now, path)
            _save(path, data)
    return {"ok": found}


def list_claims(
    model: Optional[str] = None, *, path: Optional[Path] = None,
    now_fn: Callable[[], datetime] = _default_now,
) -> list[dict]:
    """Return active claims, raising if an existing ledger cannot be trusted."""
    path = path or _DEFAULT_PATH
    target = canonical_model(model) if model is not None else None
    now = now_fn()
    data = _load(path)
    active = []
    for record in data["claims"]:
        if not _is_active(record, now):
            continue
        canonical = _canonical_record(record)
        if canonical is None:
            continue
        if target is None or canonical.get("model") == target:
            active.append(canonical)
    return active


def list_coordination(
    model: Optional[str] = None, *, path: Optional[Path] = None,
    now_fn: Callable[[], datetime] = _default_now,
) -> dict:
    """Read claims and pending operations from one consistent ledger snapshot.

    An operation is retained when its per-model OS lock is still held, even if
    its wall-clock lease has elapsed.  A released owner with an elapsed lease
    is pruned, while an uncertain result remains visible until its retry window
    ends.  This is deliberately the only reader used by the list_claims MCP
    response, so claims and operations cannot describe different ledger reads.
    """
    path = path or _DEFAULT_PATH
    target = canonical_model(model) if model is not None else None
    now = now_fn()
    with _locked(path):
        data = _load(path)
        before = (len(data["claims"]), len(data["operations"]))
        _prune_expired(data, now, path)
        if before != (len(data["claims"]), len(data["operations"])):
            _save(path, data)
        active_claims = []
        for record in data["claims"]:
            if not _is_active(record, now):
                continue
            canonical = _canonical_record(record)
            if canonical is None:
                continue
            if target is not None and canonical.get("model") != target:
                continue
            view = _claim_view(canonical)
            if view is not None:
                active_claims.append(view)
        operations = []
        for record in data["operations"]:
            try:
                operation = _operation_view(path, record, now)
            except (KeyError, TypeError, ValueError):
                continue
            if target is None or operation["model"] == target:
                operations.append(operation)
    return {"claims": active_claims, "operations": operations}


def begin_operation(
    model: str, kind: str, force: bool = False, *, path: Optional[Path] = None,
    now_fn: Callable[[], datetime] = _default_now, scope: str = _DEFAULT_SCOPE,
) -> dict:
    """Atomically reserve a same-model operation lease before backend work.

    A 120-second lease is conservative but bounded: a crashed caller can delay
    a same-model request until expiry, rather than permanently block the ledger.
    """
    path = path or _DEFAULT_PATH
    model = canonical_model(model)
    kind = nonblank_text(kind, "kind")
    scope = nonblank_text(scope, "scope")
    now = now_fn()
    with _locked(path):
        data = _load(path)
        _prune_expired(data, now, path)
        retry_counts = data.get("operation_retry_counts", {})
        if not isinstance(retry_counts, dict):
            retry_counts = {}
        retry_count = retry_counts.get(model, 0)
        if isinstance(retry_count, bool) or not isinstance(retry_count, int) or retry_count < 0:
            retry_count = 0
        pending = _pending_for(data, model, now)
        if pending:
            return {"ok": False, "outcome": "refused", "reason": "operation_pending",
                    "model": model, "operations": [_operation_view(path, r, now) for r in pending]}
        if kind == "warm":
            capacity_pending = _pending_warm_for_scope(data, scope)
            if capacity_pending:
                return {"ok": False, "outcome": "refused", "reason": "capacity_pending",
                        "model": model, "scope": scope,
                        "operations": [_operation_view(path, r, now) for r in capacity_pending]}
        active_claims = _active_model_claims(data, model, now)
        if kind in _EVICTION_KINDS and not force and active_claims:
            return {"ok": False, "outcome": "refused", "reason": "model_claimed",
                    "model": model,
                    "claims": [_claim_view(record) for record in active_claims]}
        fd = _try_operation_lock(path, model)
        if fd is None:
            return {"ok": False, "outcome": "refused", "reason": "operation_pending",
                    "model": model, "operations": []}
        try:
            operation_id = uuid.uuid4().hex
            expires_at = _expires_at(now, _OPERATION_LEASE_SECONDS)
            record = {"operation_id": operation_id, "model": model, "kind": kind,
                      "scope": scope, "started_at": _iso(now), "expires_at": _iso(expires_at),
                      "pending_until": _iso(expires_at), "lifecycle": "in_flight",
                      "outcome": None, "reason": None,
                      "retry_count": retry_count,
                      "retry_after": None}
            data["operations"].append(record)
            _save(path, data)
            _OPERATION_LOCKS[operation_id] = fd
        except Exception:
            _release_operation_lock(fd)
            raise
    # retry_count comes from the record just written, not recomputed: the lease
    # a caller is told about and the one on disk must not be able to disagree.
    return {"ok": True, "outcome": "begun", "operation_id": operation_id,
            "model": model, "kind": kind, "scope": scope,
            "expires_at": record["expires_at"], "retry_count": record["retry_count"]}


def finish_operation(
    operation_id: str, uncertain: bool = False, *, path: Optional[Path] = None,
    now_fn: Callable[[], datetime] = _default_now, reason: Optional[str] = None,
) -> dict:
    """Clear an operation lease, or retain it after unknown backend outcome.

    ``uncertain=True`` releases this process's live owner lock but preserves the
    bounded durable lease, so a timed-out request does not immediately permit a
    conflicting same-model mutation.
    """
    path = path or _DEFAULT_PATH
    operation_id = nonblank_text(operation_id, "operation_id")
    now = now_fn()
    expires_at = None
    found = False
    operation_model = None
    try:
        with _locked(path):
            data = _load(path)
            _prune_expired(data, now, path)
            found = any(isinstance(r, dict) and r.get("operation_id") == operation_id
                        for r in data["operations"])
            if found:
                for record in data["operations"]:
                    if isinstance(record, dict) and record.get("operation_id") == operation_id:
                        model = record.get("model")
                        if isinstance(model, str):
                            try:
                                operation_model = canonical_model(model)
                            except ValueError:
                                operation_model = None
                        break
            if found and uncertain:
                expires_at = _iso(_expires_at(now, _OPERATION_LEASE_SECONDS))
                for record in data["operations"]:
                    if isinstance(record, dict) and record.get("operation_id") == operation_id:
                        record["expires_at"] = expires_at
                        record["pending_until"] = expires_at
                        record["lifecycle"] = "unknown"
                        record["outcome"] = "unknown"
                        record["reason"] = reason or "outcome_unknown"
                        record["retry_after"] = expires_at
                        retry_count = record.get("retry_count", 0)
                        record["retry_count"] = retry_count if isinstance(retry_count, int) and retry_count >= 0 else 0
                _save(path, data)
            elif found:
                data["operations"] = [
                    r for r in data["operations"]
                    if not (isinstance(r, dict) and r.get("operation_id") == operation_id)
                ]
                if operation_model is not None:
                    data.setdefault("operation_retry_counts", {}).pop(operation_model, None)
                _save(path, data)
            fd = _OPERATION_LOCKS.pop(operation_id, None)
            if fd is not None:
                _release_operation_lock(fd)
    finally:
        fd = _OPERATION_LOCKS.pop(operation_id, None)
        if fd is not None:
            _release_operation_lock(fd)
    result = {"ok": found, "retained": bool(found and uncertain)}
    if found and uncertain:
        result["expires_at"] = expires_at
    return result
