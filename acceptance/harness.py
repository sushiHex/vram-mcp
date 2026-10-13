"""Opt-in acceptance run of vram-mcp against a real NVIDIA GPU and Ollama.

Never part of ``python -m pytest``. Read-only checks run by default. Scenarios
that load and unload a model run only when ``--model``, ``--ollama`` and
``--gpu`` are all given, so nothing is mutated by accident.

Every server it starts is a real ``vram-mcp`` stdio process with a fresh
temporary ``VRAM_MCP_STATE_DIR``, so the harness coordinates only with itself
and never writes the live ledger. It reads that ledger once, to refuse to touch
a model a live session has claimed. It loads only a model that was not
resident when it started, with a short keep-alive, and unloads only that model.

See docs/acceptance.md for prerequisites, scenarios and validated environments.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import subprocess
import sys
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path

import requests
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from vram_mcp import claims
from vram_mcp.models import canonical_model

KEEP_ALIVE = "2m"           # a crashed run's model still leaves VRAM on its own
EXPECTED_TOOLS = 13
CLAIM_TTL_SECONDS = 5


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


def _names(rows) -> set[str]:
    return {canonical_model(row["name"]) for row in rows or [] if row.get("name")}


def _alias(model: str) -> str:
    """A different spelling of the same model, to prove canonical matching."""
    for candidate in (f"registry.ollama.ai/library/{model.upper()}", model.upper()):
        if candidate != model and canonical_model(candidate) == canonical_model(model):
            return candidate
    return model


# ── scenarios ────────────────────────────────────────────────────────────────

async def read_only(run: Run, tools: Tools, gpu: int) -> dict:
    listed = await tools.client.list_tools()
    run.check("Every tool is discoverable", len(listed.tools) == EXPECTED_TOOLS,
              f"{len(listed.tools)} tools: {', '.join(sorted(t.name for t in listed.tools))}")

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
    run.check("The isolated ledger is read even with nothing to annotate",
              obs.get("coordination", {}).get("status") == "available",
              f"coordination: {obs.get('coordination')}")
    processes = obs.get("processes", {})
    run.check("Process observations report their coverage",
              processes.get("status") == "available" and "coverage" in processes,
              f"{len(status.get('other_processes') or [])} other process(es); "
              f"coverage {processes.get('coverage')}")
    coverage = status["pressure"]["coverage"]
    run.check("Pressure states the evidence it rests on",
              set(coverage) == {"capacity", "models", "processes", "non_local_memory"},
              f"pressure {status['pressure']['state']}, coverage {coverage}")
    return status


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
        await a("release", claim_id=held["claim_id"])


async def pending_operation_protects(run: Run, a: Tools, b: Tools, model: str) -> None:
    warming = asyncio.ensure_future(a("warm", model=model, keep_alive=KEEP_ALIVE,
                                      by="acceptance-a"))
    seen = None
    while not warming.done() and seen is None:
        listed = await b("list_claims", model=model)
        seen = next(iter(listed["operations"]), None)
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
    loaded = await b("list_loaded")
    run.check("A confirmed warm agrees with Ollama residency",
              warmed["outcome"] == "succeeded" and canonical_model(model) in _names(loaded["loaded"]),
              f"warm -> {warmed['outcome']} ({warmed['reason']}); "
              f"resident: {sorted(_names(loaded['loaded']))}")


@asynccontextmanager
async def reserving_all_but(b: Tools, headroom_mb: int):
    """Reserve every free MB but ``headroom_mb``, measured now, for the block."""
    free_mb = (await b("vram_status"))["free_mb"]
    gb = round(max(free_mb - headroom_mb, 1) / 1024, 2)
    held = await b("reserve", gb=gb, owner="acceptance-b", purpose="acceptance",
                   ttl_seconds=300)
    try:
        yield f"reserved {gb} GB of {free_mb} MB free"
    finally:
        await b("release", claim_id=held["claim_id"])


async def reservations_admit_refresh_not_reload(run: Run, a: Tools, b: Tools,
                                                model: str) -> None:
    async with reserving_all_but(b, 512) as reserved:
        refreshed = await a("warm", model=model, keep_alive=KEEP_ALIVE, by="acceptance-a")
        run.check("A resident refresh is admitted under a reservation",
                  refreshed["ok"] and refreshed["reason"] == "already_resident",
                  f"{reserved}; warm -> {refreshed['outcome']} ({refreshed['reason']})")
    unloaded = await a("unload", model=model, by="acceptance-a")
    loaded = await b("list_loaded")
    run.check("A confirmed unload agrees with Ollama residency",
              unloaded["outcome"] == "succeeded"
              and canonical_model(model) not in _names(loaded["loaded"]),
              f"unload -> {unloaded['outcome']}; resident: {sorted(_names(loaded['loaded']))}")
    async with reserving_all_but(b, 512) as reserved:
        reload = await a("warm", model=model, keep_alive=KEEP_ALIVE, by="acceptance-a")
        run.check("A reload that would breach the reservation is refused",
                  reload["outcome"] == "refused"
                  and reload["reason"] in ("no_headroom", "insufficient_headroom"),
                  f"{reserved}; warm -> {reload['outcome']} ({reload['reason']})")


async def unavailable_ollama_stays_unknown(run: Run, state_dir: Path, gpu: int,
                                           model: str) -> None:
    async with session(state_dir, "http://127.0.0.1:9", gpu) as c:
        status = await c("vram_status")
        run.check("Unreachable Ollama reads as unknown, never as empty",
                  status["loaded"] is None and status["observations"]["ollama"]["status"]
                  == "unavailable",
                  f"loaded={status['loaded']}; summary: {status['summary']}")
        warmed = await c("warm", model=model, keep_alive=KEEP_ALIVE, by="acceptance-c")
        run.check("No mutation is attempted without residency",
                  warmed["outcome"] == "refused",
                  f"warm -> {warmed['outcome']} ({warmed['reason']})")


async def claims_outlive_their_client(run: Run, state_dir: Path, ollama: str, gpu: int,
                                      b: Tools, model: str) -> None:
    async with session(state_dir, ollama, gpu) as d:
        held = await d("claim", model=model, owner="acceptance-d", purpose="acceptance",
                       ttl_seconds=CLAIM_TTL_SECONDS)
    # Session D's server process has now exited.
    refused = await b("unload", model=model, by="acceptance-b")
    await asyncio.sleep(CLAIM_TTL_SECONDS + 1)
    remaining = await b("list_claims", model=model)
    recovered = await b("unload", model=model, by="acceptance-b")
    run.check("A departed client's claim protects until its TTL, then frees the model",
              held["ok"] and refused["reason"] == "model_claimed"
              and not remaining["claims"] and recovered["outcome"] == "succeeded",
              f"after exit: unload -> {refused['reason']}; after {CLAIM_TTL_SECONDS}s TTL: "
              f"{len(remaining['claims'])} claim(s), unload -> {recovered['outcome']}")


# ── preconditions, environment, report ───────────────────────────────────────

def refuse_to_touch(model: str, ollama: str, status: dict) -> str | None:
    """Why this model must not be mutated, or None when it is safe to."""
    target = canonical_model(model)
    installed = {canonical_model(m["name"])
                 for m in requests.get(f"{ollama}/api/tags", timeout=10).json()["models"]}
    if target not in installed:
        return f"{target} is not installed"
    if target in _names(status["loaded"]):
        return f"{target} is already resident, so something else may be using it"
    if claims.list_claims(target):     # the live ledger, read only
        return f"{target} is claimed by a live session"
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
        "vram-mcp": f"{version('vram-mcp')} at "
                    f"{run_quietly('git', 'rev-parse', '--short', 'HEAD')}",
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


async def main(args: argparse.Namespace) -> int:
    run = Run()
    with tempfile.TemporaryDirectory(prefix="vram-mcp-acceptance-") as tmp:
        state = Path(tmp) / "state"
        async with session(state, args.ollama, args.gpu) as a:
            status = await read_only(run, a, args.gpu)
            env = environment(args.ollama, args.gpu, status)
            refusal = refuse_to_touch(args.model, args.ollama, status) if args.model else (
                "no --model given")
            if refusal:
                run.skip("Mutation scenarios", refusal)
            else:
                async with session(state, args.ollama, args.gpu) as b:
                    loaded_here = False
                    try:
                        await claims_protect_across_sessions(run, a, b, args.model)
                        loaded_here = True
                        await pending_operation_protects(run, a, b, args.model)
                        await reservations_admit_refresh_not_reload(run, a, b, args.model)
                        await unavailable_ollama_stays_unknown(run, Path(tmp) / "c",
                                                               args.gpu, args.model)
                        await claims_outlive_their_client(run, state, args.ollama, args.gpu,
                                                          b, args.model)
                    finally:
                        if loaded_here:     # only ever the model this run loaded
                            await b("unload", model=args.model, force=True, by="acceptance")
    text = report(env, run, args.model)
    print(text)
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
    return 1 if run.failed else 0


def parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ollama", default=None,
                        help="Ollama endpoint; required with --model")
    parser.add_argument("--gpu", type=int, default=None,
                        help="NVIDIA GPU index; required with --model")
    parser.add_argument("--model", help="an installed, non-resident model to load and unload")
    parser.add_argument("--report", help="also write the Markdown report here")
    args = parser.parse_args(argv)
    if args.model and (args.ollama is None or args.gpu is None):
        parser.error("--model mutates Ollama, so --ollama and --gpu must be given explicitly")
    args.ollama = args.ollama or "http://127.0.0.1:11434"
    args.gpu = 0 if args.gpu is None else args.gpu
    return args


if __name__ == "__main__":
    sys.exit(asyncio.run(main(parse(sys.argv[1:]))))
