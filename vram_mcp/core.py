"""Pure orchestration over a GPU-status function and an Ollama client.

Nothing here touches the network or a real GPU directly: callers inject a
``gpu_status_fn`` (``() -> list[dict]``) and an ``ollama`` client, so the whole
module is exercisable with plain fakes in tests. No ``mcp`` import.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Optional

from ._util import bytes_to_mb as _shared_bytes_to_mb
from .models import canonical_model
from .observations import Observation

_MB_PER_GB = 1024

SPILL_THRESHOLD_MB = 256   # floor only; pressure() also compares against free VRAM
TIGHT_MB = 1024

# Coverage a healthy verdict depends on, with what its absence means. Ordered:
# the first unmet requirement is the one reported, so the reason names the
# evidence nearest the reader rather than an arbitrary one. A new input that
# `ok` relies on is a line here, not another branch.
_OK_REQUIRES = (
    ("models", "Ollama residency is unavailable."),
    ("non_local_memory",
     "Non-local memory is unavailable; driver spill cannot be ruled out."),
)


def observe_loaded(ollama) -> Observation[list[dict]]:
    """Use transport health when available; plain injected clients supply data."""
    if hasattr(ollama, "observe_loaded"):
        return ollama.observe_loaded()
    return Observation(ollama.ps(), "ollama:/api/ps")


def _loaded_models(ollama=None, *, rows=None) -> list[dict]:
    """Normalize model rows; absent memory readings remain unknown."""
    loaded = []
    for m in (ollama.ps() if rows is None else rows):
        size_mb = _shared_bytes_to_mb(m.get("size"))
        vram_mb = _shared_bytes_to_mb(m.get("size_vram"))
        loaded.append(
            {
                "name": m.get("name"),
                "size_vram_mb": vram_mb,
                "total_size_mb": size_mb,
                "offloaded_to_cpu": (vram_mb < size_mb if vram_mb is not None and size_mb is not None else None),
                "expires_at": m.get("expires_at"),
            }
        )
    return loaded


class Snapshot:
    """One capture for a status view or an individual eviction decision.

    Mutations must capture again for each candidate: another session can
    register a claim while an earlier model is being unloaded.

    * ``all_claims`` — every active claim record (one ledger read).
    * ``pid_map`` — Ollama tag → runner PID (one process listing + one
      manifest walk; aliases sharing a blob all map to the runner's PID).
    * ``busy_map`` — runner PID → ``True | False | None`` (one NVML session).
    """

    def __init__(self, all_claims: list[dict], pid_map: dict,
                 busy_map: dict) -> None:
        self.all_claims = all_claims
        self.pid_map = pid_map
        self.busy_map = busy_map

    @classmethod
    def capture(cls, all_claims_fn, pid_map_fn, busy_map_fn) -> "Snapshot":
        """Run the three collectors once. ``busy_map_fn`` receives the PIDs
        the pid_map surfaced, so the NVML fetch covers exactly what's needed."""
        all_claims = all_claims_fn() if all_claims_fn else []
        pid_map = pid_map_fn() if pid_map_fn else {}
        pids = sorted(set(pid_map.values()))
        busy_map = busy_map_fn(pids) if (busy_map_fn and pids) else {}
        return cls(all_claims, pid_map, busy_map)

    def claims_for(self, model_name) -> list[dict]:
        """Active claims on ``model_name``; [] for a None/unknown name (a
        nameless ps() row must never be attributed everyone's claims)."""
        if not model_name:
            return []
        model_name = canonical_model(model_name)
        return [c for c in self.all_claims
                if isinstance(c, dict) and c.get("model")
                and canonical_model(c["model"]) == model_name]

    def pid_for(self, model_name):
        if not model_name:
            return None
        wanted = canonical_model(model_name)
        return next((pid for name, pid in self.pid_map.items()
                     if canonical_model(name) == wanted), None)

    def busy_for(self, model_name):
        pid = self.pid_for(model_name)
        if pid is None:
            return None
        return self.busy_map.get(pid)


def attach_coordination(loaded: list[dict], snap: Snapshot) -> tuple[list[dict], set]:
    """Attach ``claims`` + ``busy`` to each model dict from one Snapshot.

    Returns ``(enriched, resolved_pids)`` — ``resolved_pids`` lets callers
    exclude Ollama-runner PIDs from a general "other processes" survey, since
    they're already represented as model entries.
    """
    out = []
    resolved: set = set()
    for m in loaded:
        name = m.get("name")
        pid = snap.pid_for(name)
        if pid is not None:
            resolved.add(pid)
        out.append({**m, "claims": snap.claims_for(name), "busy": snap.busy_for(name)})
    return out, resolved


def other_processes(process_table: list[dict], exclude_pids: set) -> list[dict]:
    """Every VRAM holder in ``process_table`` that isn't an already-listed
    Ollama model.

    Takes an already-materialized table rather than the reader function on
    purpose: the caller samples the (expensive — ~1 s on Windows) source ONCE
    and feeds the same rows to both this view and ``pressure``. When this
    function called the reader itself, the filtered list was the only thing
    anyone had, and the runner's spill silently vanished from the pressure
    verdict.
    """
    return [p for p in process_table if p["pid"] not in exclude_pids]


def runner_offloads(loaded: list[dict], snap: Snapshot) -> dict[int, int]:
    """Runner PID → the MB that model DELIBERATELY placed on the CPU backend.

    This is what lets :func:`pressure` tell explained non-local memory from
    genuine paging. ``total_size_mb - size_vram_mb`` is Ollama's own account of
    the split, and ``snap.pid_for`` names the OS process the driver will report
    that memory against.

    A model whose runner PID could not be correlated entitles nobody: attributing
    its offload to an unknown PID would excuse some other process's spill.
    ``max`` rather than a sum where two tags resolve to one PID — they are
    aliases of ONE physical model (``ollama cp``, hf.co variants), so their
    offloads are the same memory counted twice, and summing would excuse double.
    """
    out: dict[int, int] = {}
    for m in loaded:
        if not isinstance(m, dict):
            continue
        pid = snap.pid_for(m.get("name"))
        if pid is None:
            continue
        offload = (m.get("total_size_mb") or 0) - (m.get("size_vram_mb") or 0)
        out[pid] = max(out.get(pid, 0), max(0, offload))
    return out


def pressure(gpus: list[dict], loaded: list[dict], process_table: list[dict],
             *, runner_offloads: Optional[dict] = None,
             spill_threshold_mb: int = SPILL_THRESHOLD_MB,
             tight_mb: int = TIGHT_MB) -> dict:
    """Assess one selected GPU using the available capacity and residency data.

    Non-local memory beyond known CPU offload, exceeding both the noise floor
    and free VRAM, suggests paging. This is a heuristic, not proof of driver
    activity. Missing capacity cannot produce a healthy verdict.
    """
    free_mb = gpus[0].get("free_mb") if gpus else None
    entitlements = runner_offloads or {}
    explained_mb = 0
    unexplained_mb = 0
    for p in process_table:
        if not isinstance(p, dict):
            continue
        non_local = p.get("non_local_mb") or 0
        if non_local <= 0:
            continue
        entitled = entitlements.get(p.get("pid"))
        if entitled is None:      # not a runner -> nothing accounts for it
            unexplained_mb += non_local
            continue
        covered = min(non_local, max(entitled, 0))
        explained_mb += covered
        unexplained_mb += non_local - covered
    non_local_mb = explained_mb + unexplained_mb
    # Non-local allocation alone does not establish memory pressure.
    spilling = unexplained_mb >= spill_threshold_mb and (
        free_mb is None or unexplained_mb > free_mb
    )
    offloaded = [
        m["name"] for m in loaded
        if isinstance(m, dict) and m.get("offloaded_to_cpu") and m.get("name")
    ]

    if spilling:
        state = "thrashing"
        detail = (
            f"{unexplained_mb} MB of unexplained non-local memory under VRAM "
            "pressure suggests paging. Free VRAM or reduce load."
        )
        # Say so, or the reader assumes the whole non-local figure is paging.
        if explained_mb:
            detail += (
                f" (A further {explained_mb} MB of non-local memory is Ollama's "
                "deliberate CPU offload, not paging.)"
            )
    elif offloaded:
        state = "degraded"
        detail = (
            f"Model(s) partly on CPU: {', '.join(offloaded)}. Slower than "
            "full GPU residency, but a deliberate Ollama placement, not paging."
        )
    elif free_mb is not None and free_mb < tight_mb:
        state = "tight"
        detail = (
            f"Only {free_mb} MB free; the next load will likely spill or fail."
        )
    elif free_mb is None:
        state = "unknown"
        detail = "GPU capacity is unavailable; pressure cannot be assessed."
    else:
        state = "ok"
        detail = "No VRAM pressure detected."

    return {
        "state": state,
        "free_mb": free_mb,
        "non_local_mb": non_local_mb,
        "explained_offload_mb": explained_mb,
        "unexplained_spill_mb": unexplained_mb,
        "spilling": spilling,
        "offloaded_models": offloaded,
        "detail": detail,
    }


def combined_status(
    gpu_status_fn: Callable[[], list[dict]], ollama, *,
    snapshot_fn=None, nvml_processes_fn=None, procinfo_fn=None,
    spill_threshold_mb: int = SPILL_THRESHOLD_MB,
) -> dict:
    """Collect a scoped status view with explicit observation health.

    GPU and process observations must describe the same device. Ollama model
    residency remains server-wide; only correlated selected-device runners
    contribute to the GPU pressure verdict. Each expensive source is read once.
    Unknown sources retain metadata and return null data, never fabricated zeros.
    """
    gpu_reading = gpu_status_fn()
    if not isinstance(gpu_reading, Observation):
        gpu_reading = Observation(gpu_reading or None, "gpu", error=None if gpu_reading else "No GPU reading")
    gpus = gpu_reading.data if gpu_reading.known else []
    # A status describes one device. Never borrow headroom from another GPU.
    gpus = gpus[:1]
    model_reading = observe_loaded(ollama)
    loaded = _loaded_models(rows=model_reading.data) if model_reading.known else []
    observations = {"gpu": gpu_reading.metadata(), "ollama": model_reading.metadata()}
    resolved_pids: set = set()
    offloads: dict = {}
    snap = None
    # Nothing loaded -> nothing to enrich; skip the snapshot's subprocess/IO.
    if snapshot_fn is not None and loaded:
        try:
            snap = snapshot_fn()
            observations["coordination"] = Observation(True, "claims+nvml").metadata()
        except (OSError, ValueError, TimeoutError) as exc:
            snap = None
            observations["coordination"] = Observation(None, "claims+nvml", error=str(exc)).metadata()
        if snap is not None:
            loaded, resolved_pids = attach_coordination(loaded, snap)
        # The SAME snapshot answers both questions, so the pid map is walked
        # once: which PIDs are runners, and how much offload each explains.
            offloads = runner_offloads(loaded, snap)
    result = {
        "gpus": gpus,
        "loaded": loaded if model_reading.known else None,
        "free_mb": gpus[0].get("free_mb") if gpus else None,
        "observations": observations,
        "scope": gpu_reading.scope,
    }
    source = procinfo_fn if procinfo_fn is not None else nvml_processes_fn
    # The source is sampled EXACTLY ONCE; on Windows it performs one bounded
    # identity lookup for the selected NVML PIDs and never samples counters.
    # the same rows feed both consumers, which need DIFFERENT views:
    #   * pressure  -> the FULL table plus the offload entitlements. The Ollama
    #     runner is normally the biggest VRAM holder and CAN genuinely be paged,
    #     so hiding it blinds spill detection; but its non-local memory is
    #     mostly its own deliberate offload, so counting it raw invents a spill.
    #     Only the full table + entitlements can tell those apart.
    #   * other_processes -> the runner-filtered view, since those PIDs are
    #     already reported as model entries.
    full_table: list[dict] = []
    if source is not None:
        process_reading = source()
        if not isinstance(process_reading, Observation):
            process_reading = Observation(process_reading, "processes")
        if process_reading.scope and gpu_reading.scope and process_reading.scope != gpu_reading.scope:
            process_reading = Observation(None, process_reading.source, error="GPU scope mismatch",
                                          scope=process_reading.scope)
        observations["processes"] = process_reading.metadata()
        full_table = process_reading.data if process_reading.known else []
        result["other_processes"] = (other_processes(full_table, resolved_pids)
                                     if process_reading.known else None)
    pressure_models = loaded
    if source is not None and gpu_reading.scope:
        selected_pids = {row["pid"] for row in full_table}
        pressure_models = [row for row in loaded if snap is not None
                           and snap.pid_for(row["name"]) in selected_pids]
    result["pressure"] = pressure(gpus, pressure_models, full_table,
                                  runner_offloads=offloads,
                                  spill_threshold_mb=spill_threshold_mb)
    non_local_known = (source is not None and process_reading.known
                       and (process_reading.coverage or {}).get("non_local_memory", True)
                       and all(p.get("non_local_mb") is not None for p in full_table))
    result["pressure"]["coverage"] = {
        "capacity": gpu_reading.known, "models": model_reading.known,
        "processes": source is not None and process_reading.known,
        "non_local_memory": non_local_known,
    }
    if not non_local_known:
        for key in ("non_local_mb", "explained_offload_mb", "unexplained_spill_mb", "spilling"):
            result["pressure"][key] = None
    # `ok` is the one verdict that asserts a NEGATIVE — that nothing is wrong —
    # so it is earned only when every input it depends on was actually read.
    # The other states rest on evidence they did observe (`tight` on capacity,
    # `degraded` on residency, `thrashing` on the non-local figures it grades),
    # so missing coverage never downgrades them; it would discard a fact.
    # `capacity` is absent from the table on purpose: pressure() already returns
    # "unknown" when free_mb is None, so a missing-capacity entry here could
    # never fire.
    coverage = result["pressure"]["coverage"]
    if result["pressure"]["state"] == "ok":
        unmet = next((why for key, why in _OK_REQUIRES if not coverage[key]), None)
        if unmet is not None:
            result["pressure"].update(state="unknown", detail=unmet)
    return result


def is_protected(model_name: str, snap: Snapshot) -> tuple[bool, dict]:
    """Is ``model_name`` unsafe to evict right now?

    Protected if EITHER an active claim exists OR its best-effort ``busy``
    signal is ``True`` — not claim-status alone, so an uncooperative caller
    that never calls ``claim()`` still can't make a real in-flight
    generation trivially interruptible. Returns ``(protected, detail)``,
    ``detail`` = ``{"claims": [...], "busy": bool | None}``.
    """
    active_claims = snap.claims_for(model_name)
    busy = snap.busy_for(model_name)
    protected = bool(active_claims) or busy is True
    return protected, {"claims": active_claims, "busy": busy}


def reserved_mb(all_claims: list[dict]) -> int:
    """Total VRAM (MB) spoken for by active reservations.

    ``all_claims`` is the ledger's already-expiry-filtered list, so every
    reservation here is live. Malformed records are skipped rather than
    raising — one unusable record must not break a status call.

    Non-positive sizes are skipped too, even though ``claims.reserve`` already
    rejects them: the ledger is a plain JSON file anyone can hand-edit, and a
    negative ``gb`` would SUBTRACT from the total — letting one record cancel
    another session's reservation. ``not (value > 0)`` also excludes NaN, which
    would otherwise poison the sum.
    """
    total = 0.0
    for record in all_claims:
        if not isinstance(record, dict) or record.get("kind") != "reservation":
            continue
        try:
            value = float(record["gb"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(value) or not value > 0:
            continue
        total += value
    return int(round(total * _MB_PER_GB))


def can_warm(model: str, *, free_mb, reserved_mb: int, model_size_mb,
             resident: bool = False) -> tuple[bool, dict]:
    """May ``model`` be warmed without eating VRAM another session reserved?

    Cooperative, not enforced: vram-mcp cannot intercept an Ollama auto-load
    triggered by a direct ``/api/generate`` call from another process, so this
    gates only vram-mcp's own ``warm()``. Every refusal is overridable with
    ``force=True``.

    Refuses when reservations leave no headroom at all, or when the model's
    approximate size exceeds the headroom. Never refuses on a guess: unknown
    free VRAM always allows.

    Returns ``(allowed, detail)`` where detail carries ``reason``,
    ``headroom_mb``, ``reserved_mb`` and ``model_size_mb``. ``reason`` never
    over-claims: ``"fits"`` means the size WAS checked against the headroom,
    ``"size_unknown"`` means it could not be, so a caller can tell a verified
    fit from an unverified one.
    """
    base = {"reserved_mb": reserved_mb, "model_size_mb": model_size_mb,
            "free_mb": free_mb, "additional_mb": 0 if resident else model_size_mb}
    if resident:
        return True, {**base, "headroom_mb": None if free_mb is None else free_mb - reserved_mb,
                      "reason": "already_resident"}
    if free_mb is None:
        return True, {**base, "headroom_mb": None, "reason": "free_unknown"}

    headroom = free_mb - reserved_mb
    detail = {**base, "headroom_mb": headroom}
    # Nothing reserved -> nobody to protect, so this predicate stays out of the
    # way even when VRAM looks tight; making room is ensure_free's job.
    if reserved_mb <= 0:
        return True, {**detail, "reason": "no_reservations"}
    if headroom <= 0:
        return False, {**detail, "reason": "no_headroom"}
    if model_size_mb is None:
        return True, {**detail, "reason": "size_unknown"}
    if model_size_mb > headroom:
        return False, {**detail, "reason": "insufficient_headroom"}
    return True, {**detail, "reason": "fits"}


def ensure_free(
    target_gb: float,
    gpu_status_fn: Callable[[], list[dict]],
    ollama,
    settle: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
    *,
    snapshot_fn=None, force: bool = False, evict_fn=None,
) -> dict:
    """Evict largest models until the selected GPU reaches the requested headroom.

    Recheck protection for each candidate. Production callers supply evict_fn
    to transact the decision and verify the result through the shared operation
    protocol. HTTP runs outside the ledger lock. Stop on unknown capacity or an
    uncertain mutation; a blind retry could interrupt another session's work.
    """
    if isinstance(target_gb, bool) or not isinstance(target_gb, (int, float)) or not math.isfinite(target_gb) or target_gb <= 0:
        raise ValueError("gb must be a positive finite number")
    if not math.isfinite(target_gb * _MB_PER_GB):
        raise ValueError("gb is too large to express as megabytes")
    target_mb = max(1, int(round(target_gb * _MB_PER_GB)))
    observations = {}

    def current_free() -> Optional[int]:
        reading = gpu_status_fn()
        if isinstance(reading, Observation):
            observations["gpu"] = reading.metadata()
            rows = reading.data if reading.known else []
        else:
            rows = reading
        return rows[0].get("free_mb") if rows else None

    free = current_free()
    if free is not None and free >= target_mb:
        return {
            "ok": True,
            "already_free": True,
            "free_mb": free,
            "unloaded": [],
            "declined": [],
            "target_mb": target_mb,
            "outcome": "succeeded", "observations": observations,
        }

    if free is None:
        return {"ok": False, "outcome": "unknown", "already_free": False,
                "free_mb": None, "unloaded": [], "declined": [], "target_mb": target_mb,
                "observations": observations, "detail": "No GPU capacity reading; no models evicted"}

    reading = observe_loaded(ollama)
    observations["ollama"] = reading.metadata()
    if not reading.known:
        return {"ok": False, "outcome": "unknown", "already_free": False,
                "free_mb": free, "unloaded": [], "declined": [], "target_mb": target_mb,
                "observations": observations, "detail": "Ollama residency unavailable; no models evicted"}

    # Largest-first so we free the most VRAM with the fewest evictions.
    models = sorted(
        _loaded_models(rows=reading.data),
        key=lambda m: m["size_vram_mb"] or 0,
        reverse=True,
    )

    unloaded: list[str] = []
    declined: list[dict] = []
    attempts: list[dict] = []
    for m in models:
        name = m["name"]
        if not name:
            continue
        if snapshot_fn is not None and not force and evict_fn is None:
            protected, detail = is_protected(name, snapshot_fn())
            if protected:
                declined.append({"name": name, **detail})
                continue
        if evict_fn is not None:
            outcome = evict_fn(name)
        else:
            ok = ollama.unload(name)
            outcome = {"model": name, "ok": ok, "outcome": "succeeded" if ok else "failed"}
        attempts.append(outcome)
        if outcome["outcome"] == "refused":
            declined.append({"name": name, **outcome})
            continue
        if outcome["ok"]:
            unloaded.append(name)
            if settle:
                sleep(settle)
        free = current_free()
        if free is None or outcome["outcome"] == "unknown" or free >= target_mb:
            break

    ok = free is not None and free >= target_mb
    return {
        "ok": ok,
        "already_free": False,
        "free_mb": free,
        "unloaded": unloaded,
        "declined": declined,
        "target_mb": target_mb,
        "outcome": ("succeeded" if ok else "unknown" if free is None or any(
            a["outcome"] == "unknown" for a in attempts) else "refused" if declined else "failed"),
        "attempts": attempts, "observations": observations,
    }


def _expires_is_forever(expires_at) -> bool:
    """True if ``expires_at`` denotes an effectively-never expiry (pinned).

    Ollama uses a far-future / zero-year timestamp for ``keep_alive=-1``.
    """
    if expires_at in (None, "", "forever"):
        return False
    text = str(expires_at)
    # Ollama emits e.g. "0001-01-01T00:00:00Z" for a never-expiring model, and
    # far-future years for very long keep-alives.
    if text.startswith("0001-01-01"):
        return True
    year = text[:4]
    if year.isdigit() and int(year) >= 9999:
        return True
    return False


def advise(gpu_status_fn: Callable[[], list[dict]], ollama) -> dict:
    """Heuristic suggestions for keeping VRAM healthy.

    Returns ``{"suggestions": [str, ...]}``. Empty list means nothing to flag.
    """
    status = combined_status(gpu_status_fn, ollama)
    loaded = status["loaded"] or []
    free_mb = status["free_mb"]

    suggestions: list[str] = []

    # Multiple models resident while free VRAM is low (or unknown).
    low_free = free_mb is None or free_mb < 2 * _MB_PER_GB
    if len(loaded) > 1 and low_free:
        suggestions.append(
            "Multiple models are loaded and free VRAM is low; set "
            "OLLAMA_MAX_LOADED_MODELS=1 to keep only one model resident."
        )

    # Any model pinned effectively forever.
    pinned = [m["name"] for m in loaded if _expires_is_forever(m["expires_at"])]
    if pinned:
        names = ", ".join(str(n) for n in pinned)
        suggestions.append(
            f"Model(s) pinned in VRAM indefinitely ({names}); set a finite "
            "OLLAMA_KEEP_ALIVE (e.g. 5m) so idle models release VRAM."
        )

    return {"suggestions": suggestions, "observations": status["observations"],
            "known": status["observations"]["gpu"]["status"] == "available"
            and status["observations"]["ollama"]["status"] == "available"}
