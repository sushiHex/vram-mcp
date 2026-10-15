"""Opt-in acceptance run of vram-mcp against a real NVIDIA GPU and Ollama.

Never part of ``python -m pytest``. Read-only checks run by default. Scenarios
that load and unload a model run only when ``--model``, ``--ollama`` and
``--gpu`` are all given, so nothing is mutated by accident. Scenarios that
overcommit VRAM, paging every process on the GPU, run only with
``--overcommit``.

Every server it starts is a real ``vram-mcp`` stdio process with a fresh
temporary ``VRAM_MCP_STATE_DIR``, so the harness coordinates only with itself
and never writes the live ledger. Mutation needs an idle local Ollama: with
nothing resident, loading the test model cannot make Ollama evict anyone
else's, and unloading it can only undo what this run did.

See docs/acceptance.md for prerequisites, scenarios and validated environments.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlparse

import requests
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from vram_mcp import __version__, claims
from vram_mcp.models import canonical_model

KEEP_ALIVE = "2m"       # an interrupted run's model still leaves VRAM on its own
CLAIM_TTL_SECONDS = 5
MIN_MODEL_MB = 64       # reservation rounding must never leave room for the model
EXPECTED_TOOLS = {"advise", "claim", "ensure_free", "history", "list_claims", "list_loaded",
                  "release", "renew", "reserve", "trend", "unload", "vram_status", "warm"}
LOOPBACK = {"127.0.0.1", "localhost", "::1"}


# ── sessions ─────────────────────────────────────────────────────────────────

class ToolError(RuntimeError):
    pass


class Tools:
    """Call a tool on one MCP session and return its structured result."""

    def __init__(self, client: ClientSession) -> None:
        self.client = client

    async def __call__(self, name: str, **arguments) -> dict:
        result = await self.client.call_tool(name, arguments)
        if result.isError:
            raise ToolError(f"{name}: {result.content[0].text if result.content else 'error'}")
        data = result.structuredContent
        if data is None:
            data = json.loads(result.content[0].text)
        # FastMCP wraps a bare-dict return as {"result": ...}.
        return data["result"] if list(data) == ["result"] else data

    async def resident(self) -> set[str]:
        """Canonical names Ollama has resident. Unknown residency is an error,
        never an empty set: an empty set would pass every "is it gone" check."""
        loaded = (await self("list_loaded"))["loaded"]
        if loaded is None:
            raise ToolError("Ollama residency is unavailable")
        return {canonical_model(row["name"]) for row in loaded if row.get("name")}


@asynccontextmanager
async def session(state_dir: Path, ollama: str, gpu: int):
    """One independent vram-mcp server process, as an MCP client launches it."""
    env = {**os.environ, "VRAM_MCP_STATE_DIR": str(state_dir),
           "OLLAMA_BASE_URL": ollama, "VRAM_MCP_GPU_INDEX": str(gpu)}
    params = StdioServerParameters(command=sys.executable,
                                   args=["-m", "vram_mcp.server"], env=env)
    with open(os.devnull, "w") as quiet:      # per-request INFO logging
        async with stdio_client(params, errlog=quiet) as (read, write), \
                ClientSession(read, write) as client:
            await client.initialize()
            yield Tools(client)


# RFC 6761 reserves .invalid: it never resolves, so no server can ever answer.
UNREACHABLE = "http://ollama.invalid:11434"


# ── results ──────────────────────────────────────────────────────────────────

@dataclass
class Run:
    results: list[tuple[str, str, str]] = field(default_factory=list)

    def check(self, scenario: str, passed: bool, evidence: str) -> bool:
        self.results.append((scenario, "pass" if passed else "FAIL", evidence))
        return passed

    def skip(self, scenario: str, reason: str) -> None:
        self.results.append((scenario, "skipped", reason))

    @property
    def failed(self) -> bool:
        return any(outcome == "FAIL" for _, outcome, _ in self.results)


def _alias(model: str) -> str:
    """A different spelling of the same model, to prove canonical matching."""
    for candidate in (f"registry.ollama.ai/library/{model.upper()}", model.upper()):
        if candidate != model and canonical_model(candidate) == canonical_model(model):
            return candidate
    return model


# ── read-only scenarios ──────────────────────────────────────────────────────

async def read_only(run: Run, tools: Tools, gpu: int) -> dict:
    listed = {tool.name for tool in (await tools.client.list_tools()).tools}
    run.check("Every tool is discoverable", listed == EXPECTED_TOOLS,
              f"{len(listed)} tools; missing {sorted(EXPECTED_TOOLS - listed)}, "
              f"unexpected {sorted(listed - EXPECTED_TOOLS)}")

    status = await tools("vram_status")
    obs = status["observations"]
    gpu_obs = obs["gpu"]
    run.check("The selected GPU is identified with provenance",
              gpu_obs["status"] == "available" and gpu_obs["scope"] == f"gpu:index={gpu}"
              and bool(status["gpus"]) and status["free_mb"] is not None,
              f"{gpu_obs['source']} at {gpu_obs['observed_at']}, scope {gpu_obs['scope']}, "
              f"free {status['free_mb']} MB")
    run.check("Ollama residency is observed",
              obs["ollama"]["status"] == "available" and status["loaded"] is not None,
              f"{obs['ollama']['source']}: {len(status['loaded'] or [])} model(s) loaded")
    coordination = obs.get("coordination", {})
    expected_source = "claims+nvml" if status["loaded"] else "claims"
    run.check("The ledger is read and reported, loaded models or not",
              coordination.get("status") == "available"
              and coordination.get("source") == expected_source,
              f"{coordination.get('source')}: {coordination.get('status')}")
    processes = obs.get("processes", {})
    process_coverage = processes.get("coverage") or {}
    run.check("Process observations report their coverage",
              processes.get("status") == "available" and {
                  "compute_processes", "graphics_processes", "non_local_memory"
              } <= set(process_coverage),
              f"{len(status.get('other_processes') or [])} other process(es); "
              f"coverage {process_coverage}")
    coverage = status["pressure"]["coverage"]
    run.check("Pressure states the evidence it rests on",
              set(coverage) == {"capacity", "models", "processes", "non_local_memory",
                                "paging"},
              f"pressure {status['pressure']['state']}, coverage {coverage}")
    if sys.platform == "win32":
        run.check("Windows reads the selected adapter's non-local memory and "
                  "paging, so pressure reaches a verdict",
                  coverage["non_local_memory"] and coverage["paging"]
                  and status["pressure"]["state"] != "unknown",
                  f"pressure {status['pressure']['state']}: {status['pressure']['detail']}")
    else:
        run.check("Uncovered paging says why",
                  coverage["paging"] or "paging" in status["pressure"]["detail"],
                  f"paging covered: {coverage['paging']}; {status['pressure']['detail']}")
    return status


# ── mutation scenarios, in order: absent → resident → absent ────────────────

async def claims_protect_across_sessions(run: Run, a: Tools, b: Tools, model: str) -> None:
    alias = _alias(model)
    held = await a("claim", model=alias, owner="acceptance-a", purpose="acceptance",
                   ttl_seconds=300)
    try:
        run.check("A claim made under an alias names the canonical model",
                  held["ok"] and held["model"] == canonical_model(model),
                  f"claim({alias!r}) -> {held['model']!r}")
        refused = await b("unload", model=model, by="acceptance-b")
        run.check("Another session cannot evict a claimed model",
                  refused["outcome"] == "refused" and refused["reason"] == "model_claimed",
                  f"unload -> {refused['outcome']} ({refused['reason']})")
    finally:
        if held["ok"]:
            await a("release", claim_id=held["claim_id"])


async def pending_warm_protects(run: Run, a: Tools, b: Tools, model: str) -> None:
    warming = asyncio.ensure_future(a("warm", model=model, keep_alive=KEEP_ALIVE,
                                      by="acceptance-a"))
    seen = None
    while not warming.done() and seen is None:
        seen = next(iter((await b("list_claims", model=model))["operations"]), None)
        await asyncio.sleep(0.02)
    if seen is None:
        run.skip("A pending warm blocks other sessions",
                 "the warm completed before a second session could observe it")
    else:
        unload = await b("unload", model=model, by="acceptance-b")
        claim = await b("claim", model=model, owner="acceptance-b", purpose="acceptance")
        if claim["ok"]:     # a wrongly admitted claim must not leak into later scenarios
            await b("release", claim_id=claim["claim_id"])
        run.check("A pending warm blocks other sessions",
                  unload["reason"] == "operation_pending" and not claim["ok"],
                  f"seen {seen['kind']} {seen['lifecycle']} (owner_live={seen['owner_live']}); "
                  f"unload -> {unload['reason']}; claim -> ok={claim['ok']}")
    warmed = await warming
    resident = await b.resident()
    run.check("A confirmed warm agrees with Ollama residency",
              warmed["outcome"] == "succeeded" and canonical_model(model) in resident,
              f"warm -> {warmed['outcome']} ({warmed['reason']}); resident: {sorted(resident)}")


@asynccontextmanager
async def reserving_all_but(b: Tools, headroom_mb: int):
    """Reserve every free MB but ``headroom_mb``, measured now, for the block."""
    free_mb = (await b("vram_status"))["free_mb"]
    # Rounded down, so the headroom left is never smaller than asked for.
    gb = math.floor((free_mb - headroom_mb) / 1024 * 100) / 100
    held = await b("reserve", gb=gb, owner="acceptance-b", purpose="acceptance",
                   ttl_seconds=300)
    if not held["ok"]:
        raise ToolError(f"reserve({gb}) was refused: {held.get('summary')}")
    try:
        yield f"reserved {gb} GB of {free_mb} MB free"
    finally:
        await b("release", claim_id=held["claim_id"])


async def refresh_is_admitted_under_reservation(run: Run, a: Tools, b: Tools,
                                                model: str) -> None:
    # Everything free is reserved: a refresh needs no new residency, so even
    # a full reservation must admit it.
    async with reserving_all_but(b, 0) as reserved:
        refreshed = await a("warm", model=model, keep_alive=KEEP_ALIVE, by="acceptance-a")
        run.check("A resident refresh is admitted under a full reservation",
                  refreshed["ok"] and refreshed["reason"] == "already_resident",
                  f"{reserved}; warm -> {refreshed['outcome']} ({refreshed['reason']})")


async def claims_outlive_their_client(run: Run, state: Path, ollama: str, gpu: int,
                                      b: Tools, model: str) -> None:
    async with session(state, ollama, gpu) as d:
        held = await d("claim", model=model, owner="acceptance-d", purpose="acceptance",
                       ttl_seconds=CLAIM_TTL_SECONDS)
    # Session D's server process has now exited, leaving its claim behind.
    refused = await b("unload", model=model, by="acceptance-b")
    await asyncio.sleep(CLAIM_TTL_SECONDS + 1)
    remaining = (await b("list_claims", model=model))["claims"]
    unloaded = await b("unload", model=model, by="acceptance-b")
    resident = await b.resident()
    run.check("A departed client's claim protects until its TTL",
              held["ok"] and refused["reason"] == "model_claimed" and not remaining,
              f"after exit: unload -> {refused['reason']}; "
              f"after {CLAIM_TTL_SECONDS}s: {len(remaining)} claim(s)")
    run.check("A confirmed unload agrees with Ollama residency",
              unloaded["outcome"] == "succeeded" and canonical_model(model) not in resident,
              f"unload -> {unloaded['outcome']}; resident: {sorted(resident)}")


async def reload_is_refused_under_reservation(run: Run, a: Tools, b: Tools, model: str,
                                              headroom_mb: int) -> None:
    async with reserving_all_but(b, headroom_mb) as reserved:
        reload = await a("warm", model=model, keep_alive=KEEP_ALIVE, by="acceptance-a")
        run.check("A reload that would breach a reservation is refused",
                  reload["outcome"] == "refused" and reload["reason"] == "insufficient_headroom",
                  f"{reserved}; warm -> {reload['outcome']} ({reload['reason']})")


async def unreachable_ollama_stays_unknown(run: Run, state: Path, gpu: int, model: str) -> None:
    async with session(state, UNREACHABLE, gpu) as c:
        status = await c("vram_status")
        run.check("Unreachable Ollama reads as unknown, never as empty",
                  status["loaded"] is None
                  and status["observations"]["ollama"]["status"] == "unavailable",
                  f"loaded={status['loaded']}; summary: {status['summary']}")
        warmed = await c("warm", model=model, keep_alive=KEEP_ALIVE, by="acceptance-c")
        run.check("No mutation is attempted without residency",
                  warmed["outcome"] == "refused" and warmed["reason"] == "observation_unavailable",
                  f"warm -> {warmed['outcome']} ({warmed['reason']})")


async def unload_test_model(run: Run, b: Tools, model: str) -> None:
    """The only cleanup: the test model, which was not resident at the start."""
    result = await b("unload", model=model, force=True, by="acceptance")
    run.check("The test model is unloaded afterwards", result["outcome"] == "succeeded",
              f"unload -> {result['outcome']}")


# ── overcommit: VRAM pages, and pressure must say so ─────────────────────────

class Holder:
    """One ``overcommit.py`` process holding and touching GPU memory, and the
    lines it printed, each with the monotonic time it arrived."""

    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self.process = process
        self.lines: list[tuple[float, dict]] = []
        self._first = asyncio.Event()
        self._reader = asyncio.ensure_future(self._read())

    @classmethod
    async def start(cls, bus_id: str, mb: int, seconds: float) -> "Holder":
        return cls(await asyncio.create_subprocess_exec(
            sys.executable, str(OVERCOMMIT), bus_id, str(mb), str(seconds),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL))

    async def _read(self) -> None:
        async for line in self.process.stdout:
            self.lines.append((time.monotonic(), json.loads(line)))
            self._first.set()
        self._first.set()

    async def allocated(self) -> str | None:
        """None once its memory is allocated; otherwise why it is not."""
        await self._first.wait()
        first = self.lines[0][1] if self.lines else {"error": "it exited silently"}
        return first.get("error") if "allocated_mb" not in first else None

    def rates(self) -> list[float]:
        return [line["gbps"] for _, line in self.lines if "gbps" in line]

    def slow_spans(self, below_gbps: float) -> list[tuple[float, float]]:
        """When each pass slower than ``below_gbps`` ran, in monotonic time."""
        return [(at - line["seconds"], at) for at, line in self.lines
                if "gbps" in line and line["gbps"] < below_gbps]

    async def stop(self) -> None:
        if self.process.returncode is None:
            self.process.kill()
        await self.process.wait()
        await self._reader


def covered_seconds(spans: list[tuple[float, float]], start: float, end: float) -> float:
    """How much of ``[start, end]`` the spans cover, counting overlaps once."""
    covered, reached = 0.0, start
    for low, high in sorted((max(low, start), min(high, end)) for low, high in spans):
        if low < high and high > reached:
            covered += high - max(low, reached)
            reached = high
    return covered


OVERCOMMIT = Path(__file__).with_name("overcommit.py")
FULL_HEADROOM_MB = 256    # the first holder fills VRAM but fits: full, not paging
OVERCOMMIT_MB = 6144      # past free VRAM, by more than idle desktop memory WDDM can evict
HOLD_SECONDS = 30
PAGING_RATIO = 0.1        # a pass at a tenth of solo bandwidth went over PCIe
PAGING_WINDOW_SECONDS = 10    # the server's paging window


def refuse_to_overcommit(status: dict) -> str | None:
    """Why the overcommit scenarios must not run, or None when they may."""
    if sys.platform != "win32":
        return ("only WDDM pages ordinary CUDA allocations to system memory; "
                "see #27 for Linux")
    if status["pressure"]["state"] != "ok":
        return f"pressure already reads {status['pressure']['state']}"
    if status["loaded"]:
        return (f"Ollama has {len(status['loaded'])} model(s) resident, which "
                "overcommitting would page too")
    if status["observations"]["processes"].get("device") is None:
        return "the GPU's PCI address is unknown"
    if status["free_mb"] < 4 * FULL_HEADROOM_MB:
        return f"only {status['free_mb']} MB is free"
    return None


def _non_local(status: dict, holders: list[Holder]) -> str:
    by_pid = {row["pid"]: row["non_local_mb"] for row in status.get("other_processes") or []}
    return ", ".join(f"{by_pid.get(holder.process.pid)} MB" for holder in holders)


async def overcommit_reads_as_thrashing(run: Run, tools: Tools) -> None:
    status = await tools("vram_status")
    refusal = refuse_to_overcommit(status)
    if refusal:
        run.skip("Overcommit scenarios", refusal)
        return
    bus_id = status["observations"]["processes"]["device"]
    holders = [await Holder.start(bus_id, status["free_mb"] - FULL_HEADROOM_MB,
                                  HOLD_SECONDS + 15)]
    try:
        if why := await holders[0].allocated():
            run.skip("Overcommit scenarios", f"the first holder could not allocate: {why}")
            return
        await asyncio.sleep(5)
        full = (await tools("vram_status"))["pressure"]
        run.check("Full VRAM that is not overcommitted is not thrashing",
                  full["state"] != "thrashing",
                  f"{full['state']}: {full['free_mb']} MB free, {full['non_local_mb']} MB "
                  "non-local")
        solo = statistics.median(holders[0].rates()[1:] or [0.0])  # the first pass faults pages in
        holders.append(await Holder.start(bus_id, OVERCOMMIT_MB, HOLD_SECONDS))
        if why := await holders[1].allocated():
            run.skip("Overcommit scenarios", f"the driver refused to overcommit: {why}")
            return
        await asyncio.sleep(HOLD_SECONDS / 2)
        read_at = time.monotonic()
        during = await tools("vram_status")
    finally:
        for holder in holders:
            await holder.stop()

    # Judge paging over the window the verdict reads. One slow pass only shows
    # an eviction; WDDM can absorb an overcommit by evicting idle desktop
    # memory once, and then nothing pages.
    # Slow passes are long ones, so the measure is the share of the window
    # some holder spent in one, not how many ended in it.
    spans = [span for holder in holders for span in holder.slow_spans(solo * PAGING_RATIO)]
    slow_seconds = covered_seconds(spans, read_at - PAGING_WINDOW_SECONDS, read_at)
    paging = run.check(
        "Overcommitted VRAM keeps paging",
        slow_seconds >= PAGING_WINDOW_SECONDS / 2,
        f"for {slow_seconds:.1f} of the {PAGING_WINDOW_SECONDS} s before the reading, "
        f"a holder was in a pass under a tenth of its {solo:.0f} GB/s")
    pressure = during["pressure"]
    evidence = (f"{pressure['state']}: paging {pressure['paging']}; {pressure['free_mb']} MB "
                f"free, {pressure['non_local_mb']} MB non-local, the holders' share "
                f"{_non_local(during, holders)}")
    if paging:
        run.check("Pressure reads thrashing while VRAM pages",
                  pressure["state"] == "thrashing", evidence)
    else:
        run.skip("Pressure reads thrashing while VRAM pages",
                 f"VRAM was not paging when it was read; {evidence}")
    for _ in range(10):     # WDDM takes a moment to give the memory back
        after = (await tools("vram_status"))["pressure"]
        if after["state"] == "ok":
            break
        await asyncio.sleep(2)
    run.check("Pressure reads ok once paging stops", after["state"] == "ok",
              f"{after['state']}: {after['free_mb']} MB free, {after['non_local_mb']} MB "
              "non-local")


# ── preconditions, environment, report ───────────────────────────────────────

def refuse_to_touch(args: argparse.Namespace, run: Run, status: dict) -> str | None:
    """Why the mutation scenarios must not run, or None when they may."""
    if run.failed:
        return "a read-only check failed"
    if urlparse(args.ollama).hostname not in LOOPBACK:
        return "--ollama is not local, so its GPU cannot be the one measured here"
    if status["loaded"] is None:
        return "Ollama residency is unknown"
    if status["loaded"]:
        return (f"Ollama is not idle ({len(status['loaded'])} model(s) resident); loading "
                "the test model could make Ollama evict one")
    target = canonical_model(args.model)
    size_mb = installed_size_mb(args.ollama, target)
    if size_mb is None:
        return f"{target} is not installed"
    if not MIN_MODEL_MB <= size_mb < status["free_mb"]:
        return (f"{target} is {size_mb} MB; the reservation scenarios need a model of at "
                f"least {MIN_MODEL_MB} MB that fits in the {status['free_mb']} MB free")
    live_claims, live_operations = live_coordination()
    if live_claims or live_operations:
        return (f"the live ledger holds {len(live_claims)} claim(s) or reservation(s) and "
                f"{len(live_operations)} operation record(s); run when no session is "
                "coordinating")
    return None


def live_coordination() -> tuple[list, list]:
    """Active claims and every operation record in the live ledger, read only.

    An expired operation can still have a live owner, and telling which would
    mean taking its lock, so every record counts. ``list_coordination`` would
    prune and save, so the records are read directly.
    """
    return claims.list_claims(), claims._load(claims._DEFAULT_PATH)["operations"]


def installed_size_mb(ollama: str, model: str) -> int | None:
    for row in requests.get(f"{ollama}/api/tags", timeout=10).json()["models"]:
        if canonical_model(row["name"]) == canonical_model(model):
            return int(row["size"]) // 2**20
    return None


def environment(ollama: str, gpu: int, status: dict) -> dict:
    def run_quietly(*cmd: str) -> str:
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=15,
                                  check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return "unavailable"

    try:
        ollama_version = requests.get(f"{ollama}/api/version", timeout=10).json()["version"]
    except (requests.RequestException, ValueError, KeyError):
        ollama_version = "unavailable"
    return {
        "OS": platform.platform(),
        "Python": platform.python_version(),
        "vram-mcp": __version__ + (
            "" if (revision := run_quietly("git", "rev-parse", "--short", "HEAD")) == "unavailable"
            else f" at {revision}"),
        "mcp": version("mcp"),
        "GPU": run_quietly("nvidia-smi", f"--id={gpu}", "--format=csv,noheader",
                           "--query-gpu=name,driver_version,memory.total"),
        "Ollama": ollama_version,
        "Pressure coverage": json.dumps(status["pressure"]["coverage"]),
    }


def report(env: dict, run: Run, model: str | None) -> str:
    lines = ["# vram-mcp acceptance report", "",
             "| | |", "| --- | --- |", *(f"| {k} | {v} |" for k, v in env.items()),
             f"| Test model | {model or 'none (read-only run)'} |", "",
             "| Scenario | Outcome | Evidence |", "| --- | --- | --- |"]
    lines += [f"| {s} | {o} | {e.replace('|', '/')} |" for s, o, e in run.results]
    return "\n".join(lines) + "\n"


async def acceptance(run: Run, args: argparse.Namespace, tmp: Path) -> dict:
    state = tmp / "state"
    async with session(state, args.ollama, args.gpu) as a:
        status = await read_only(run, a, args.gpu)
        env = environment(args.ollama, args.gpu, status)
        refusal = refuse_to_touch(args, run, status) if args.model else "no --model given"
        if refusal:
            run.skip("Mutation scenarios", refusal)
        else:
            await mutation(run, args, state, tmp, a)
        if not args.overcommit:
            run.skip("Overcommit scenarios", "no --overcommit given")
        elif run.failed:
            run.skip("Overcommit scenarios", "an earlier check failed")
        else:
            await overcommit_reads_as_thrashing(run, a)
    return env


async def mutation(run: Run, args: argparse.Namespace, state: Path, tmp: Path,
                   a: Tools) -> None:
    # Half the model's size: a reload must be refused, whatever the model.
    headroom_mb = installed_size_mb(args.ollama, args.model) // 2
    async with session(state, args.ollama, args.gpu) as b:
        try:
            await claims_protect_across_sessions(run, a, b, args.model)
            await pending_warm_protects(run, a, b, args.model)
            await refresh_is_admitted_under_reservation(run, a, b, args.model)
            await claims_outlive_their_client(run, state, args.ollama, args.gpu, b,
                                              args.model)
            await reload_is_refused_under_reservation(run, a, b, args.model, headroom_mb)
            await unreachable_ollama_stays_unknown(run, tmp / "unreachable", args.gpu,
                                                   args.model)
        finally:
            await unload_test_model(run, b, args.model)


async def main(args: argparse.Namespace) -> int:
    run = Run()
    env: dict = {}
    with tempfile.TemporaryDirectory(prefix="vram-mcp-acceptance-") as tmp:
        try:
            env = await acceptance(run, args, Path(tmp))
        except Exception as exc:        # still report what did run
            traceback.print_exc()       # the report holds one line; this holds the cause
            run.check("The run completed", False, f"{type(exc).__name__}: {exc}")
    text = report(env, run, args.model)
    print(text)
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
    return 1 if run.failed else 0


def parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ollama", help="local Ollama endpoint; required with --model")
    parser.add_argument("--gpu", type=int, help="NVIDIA GPU index; required with --model")
    parser.add_argument("--model", help="an installed model to load and unload; Ollama "
                                        "must be idle")
    parser.add_argument("--overcommit", action="store_true",
                        help="also overcommit VRAM for about a minute, to check that "
                             "paging reads as thrashing; it pages every GPU user, so "
                             "run it on an otherwise idle GPU")
    parser.add_argument("--report", help="also write the Markdown report here")
    args = parser.parse_args(argv)
    if args.model and (args.ollama is None or args.gpu is None):
        parser.error("--model mutates Ollama, so --ollama and --gpu must be given explicitly")
    args.ollama = args.ollama or "http://127.0.0.1:11434"
    args.gpu = 0 if args.gpu is None else args.gpu
    return args


if __name__ == "__main__":
    sys.exit(asyncio.run(main(parse(sys.argv[1:]))))
