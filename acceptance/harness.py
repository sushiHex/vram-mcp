"""Opt-in acceptance run of vram-mcp against a real NVIDIA GPU and Ollama.

Never part of ``python -m pytest``. Read-only checks run by default. Scenarios
that load and unload a model run only when ``--model``, ``--ollama`` and
``--gpu`` are all given, so nothing is mutated by accident.

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
import subprocess
import sys
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlparse

import requests
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from vram_mcp import claims
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
              set(coverage) == {"capacity", "models", "processes", "non_local_memory"},
              f"pressure {status['pressure']['state']}, coverage {coverage}")
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
        "vram-mcp": version("vram-mcp") + (
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
            return env
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
    return env


async def main(args: argparse.Namespace) -> int:
    run = Run()
    env: dict = {}
    with tempfile.TemporaryDirectory(prefix="vram-mcp-acceptance-") as tmp:
        try:
            env = await acceptance(run, args, Path(tmp))
        except Exception as exc:        # still report what did run
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
    parser.add_argument("--report", help="also write the Markdown report here")
    args = parser.parse_args(argv)
    if args.model and (args.ollama is None or args.gpu is None):
        parser.error("--model mutates Ollama, so --ollama and --gpu must be given explicitly")
    args.ollama = args.ollama or "http://127.0.0.1:11434"
    args.gpu = 0 if args.gpu is None else args.gpu
    return args


if __name__ == "__main__":
    sys.exit(asyncio.run(main(parse(sys.argv[1:]))))
