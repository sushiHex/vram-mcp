"""Map Ollama model tags to their ``llama-server`` runner PIDs.

Ollama spawns one ``llama-server`` OS subprocess per loaded model but exposes
no PID in its own API (``/api/ps`` has no ``pid`` field, confirmed against
Ollama 0.31.1). The only correlation path: scan the runner's command line for
blob-file digests (files named ``sha256-<64 hex>``), then search Ollama's
on-disk manifests for tags whose model-layer digest matches.

Two real-world wrinkles this design absorbs:

* A runner cmdline can contain **multiple** blob paths (``--model`` plus
  ``--mmproj`` for multimodal), and paths can contain spaces/quoting. Rather
  than parsing the ``--model`` argument, every sha256 digest found anywhere in
  the cmdline is tried — only model-layer digests exist in manifests, so a
  projector blob simply never matches.
* One blob digest can belong to **multiple** tags (``ollama cp``, re-tags,
  hf.co variants — verified live: 6 digest groups spanning 15 manifests). A
  single manifest walk builds digest → {all tags}, so any alias resolves.

This is inherently undocumented and version-dependent. Every function
degrades to ``None``/``{}`` on any parse/lookup failure rather than guessing,
so a future Ollama layout change only turns a model's ``busy`` signal into
"unknown," never breaks anything else.

Because the correlation lives or dies on getting Ollama's TAG NAMING right,
this module also owns those naming rules as pure helpers — manifest path → tag
name (:func:`_tag_name_from_manifest_parts`) and bare name → ``:latest``
(:func:`resolve_tag`) — so no caller has to re-derive them inline.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

from . import winproc
from ._util import run_capture

_BLOB_DIGEST_RE = re.compile(r"sha256-([0-9a-f]{64})", re.IGNORECASE)
_MODEL_LAYER_MEDIA_TYPE = "application/vnd.ollama.image.model"
_OFFICIAL_REGISTRY = "registry.ollama.ai"


def _default_manifests_root() -> Path:
    models_dir = os.environ.get("OLLAMA_MODELS")
    if models_dir:
        return Path(models_dir) / "manifests"
    return Path.home() / ".ollama" / "models" / "manifests"


def _list_llama_server_processes_windows() -> list[dict]:
    """``[{"pid": int, "cmdline": str}, ...]`` via :mod:`.winproc`. A runner
    whose command line Windows denies is skipped; ``[]`` on any failure."""
    try:
        out = []
        for pid, name in winproc.process_names().items():
            if name.lower() != "llama-server.exe":
                continue
            cmdline = winproc.command_line(pid)
            if cmdline is not None:
                out.append({"pid": pid, "cmdline": cmdline})
        return out
    except OSError:
        return []


def _list_llama_server_processes_posix(timeout: int = 5) -> list[dict]:
    """``[{"pid": int, "cmdline": str}, ...]`` via ``ps``. ``[]`` on failure."""
    stdout = run_capture(["ps", "-eo", "pid,args"], timeout)
    if stdout is None:
        return []
    out = []
    for line in stdout.splitlines()[1:]:
        line = line.strip()
        if "llama-server" not in line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        out.append({"pid": int(parts[0]), "cmdline": parts[1]})
    return out


def _list_llama_server_processes(timeout: int = 5) -> list[dict]:
    if sys.platform == "win32":
        return _list_llama_server_processes_windows()
    return _list_llama_server_processes_posix(timeout)


def _extract_blob_digests(cmdline: str) -> list[str]:
    """Every sha256 blob digest found anywhere in a runner's command line.

    Deliberately ignores argument structure (``--model``/``--mmproj``,
    quoting, paths with spaces): only exact 64-hex digests are matched, and a
    non-model digest simply never appears in a manifest's model layer, so
    returning all of them is both simpler and immune to quoting issues.
    Lowercased; order preserved; duplicates removed.
    """
    seen: dict[str, None] = {}
    for match in _BLOB_DIGEST_RE.findall(cmdline):
        seen.setdefault(match.lower())
    return list(seen)


def _tag_name_from_manifest_parts(registry_host: str, namespace: str,
                                  name: str, tag: str) -> str:
    """The tag name Ollama itself reports in ``/api/ps`` / ``/api/tags``.

    * ``registry.ollama.ai`` + ``library`` → ``name:tag``
    * ``registry.ollama.ai`` + other namespace → ``namespace/name:tag``
    * any other registry host (e.g. ``hf.co``) → ``host/namespace/name:tag``
    """
    if registry_host == _OFFICIAL_REGISTRY:
        if namespace == "library":
            return f"{name}:{tag}"
        return f"{namespace}/{name}:{tag}"
    return f"{registry_host}/{namespace}/{name}:{tag}"


def resolve_tag(model: str, known) -> Optional[str]:
    """The key in ``known`` that Ollama would resolve ``model`` to, or ``None``.

    Ollama's naming rule: a tag-less name means the ``:latest`` tag, so
    ``ollama run llama3.2`` and ``llama3.2:latest`` are the same model. Anything
    keyed by the names Ollama reports (``/api/tags``, ``/api/ps``) therefore
    misses on a bare name — verified live: ``tags()['qwen3:32b']`` exists while
    ``tags().get('qwen3')`` is ``None``. That miss reads as "size unknown",
    which is how a bare name slipped past ``warm()``'s headroom refusal.

    A name that ALREADY carries a tag never falls back to ``:latest``:
    ``qwen3:32b`` and ``qwen3:latest`` are different models of different sizes,
    and a wrong size is worse than an unknown one.

    "Already tagged" is decided from the part after the LAST ``/``, the only
    place a tag can legally appear — the same name shape
    :func:`_tag_name_from_manifest_parts` builds. A colon earlier in the name
    belongs to a registry host's port (``localhost:5000/library/foo``), and
    reading that as a tag skipped the ``:latest`` fallback for every
    self-hosted-registry model.
    """
    if not model:
        return None
    if model in known:
        return model
    if ":" not in model.rsplit("/", 1)[-1]:
        latest = f"{model}:latest"
        if latest in known:
            return latest
    return None


def _digest_to_tags(manifests_root: Path) -> dict[str, set[str]]:
    """One manifest-tree walk → model-layer digest → all tag names using it.

    A manifest's path relative to ``manifests_root`` is always
    ``<registry-host>/<namespace>/<name>/<tag>`` — Ollama's on-disk layout
    (e.g. ``registry.ollama.ai/library/qwen3/8b``, ``hf.co/NousResearch/
    Hermes-4.3-36B-GGUF/q4_K_M``, both verified live). A path that isn't
    exactly 4 levels deep is skipped, not guessed at. ``{}`` on any failure.
    """
    mapping: dict[str, set[str]] = {}
    try:
        if not manifests_root.is_dir():
            return {}
        for manifest_path in manifests_root.rglob("*"):
            if not manifest_path.is_file():
                continue
            rel = manifest_path.relative_to(manifests_root).parts
            if len(rel) != 4:
                continue
            try:
                doc = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(doc, dict):
                continue
            layers = doc.get("layers", [])
            if not isinstance(layers, list):
                continue
            tag_name = _tag_name_from_manifest_parts(*rel)
            for layer in layers:
                if not isinstance(layer, dict):
                    continue
                if layer.get("mediaType") != _MODEL_LAYER_MEDIA_TYPE:
                    continue
                digest = layer.get("digest")
                if not isinstance(digest, str) or not digest.startswith("sha256:"):
                    continue
                hex_digest = digest[len("sha256:"):].lower()
                mapping.setdefault(hex_digest, set()).add(tag_name)
    except OSError:
        return {}
    return mapping


def runner_pid_map(*, list_processes=None, manifests_root: Optional[Path] = None) -> dict[str, int]:
    """Map every Ollama model tag currently served by a llama-server runner to that runner's PID.
    ONE process listing + ONE manifest walk. A digest shared by several tags maps ALL those tags
    to the runner's PID (any alias the caller asks about matches). {} on any failure."""
    try:
        list_processes = list_processes or _list_llama_server_processes
        manifests_root = manifests_root or _default_manifests_root()
        processes = list_processes()
        if not processes:
            return {}
        digest_tags = _digest_to_tags(manifests_root)
        if not digest_tags:
            return {}
        result: dict[str, int] = {}
        for proc in processes:
            pid = proc.get("pid")
            cmdline = proc.get("cmdline")
            if not isinstance(pid, int) or not isinstance(cmdline, str):
                continue
            for digest in _extract_blob_digests(cmdline):
                for tag in digest_tags.get(digest, ()):
                    result[tag] = pid
        return result
    except Exception:
        return {}


def find_pid_for_model(
    model_name: str, *,
    list_processes=None,
    manifests_root: Optional[Path] = None,
) -> Optional[int]:
    """The OS PID of the ``llama-server`` runner currently serving ``model_name``.

    Kept as public convenience API for single-model lookups; vram-mcp's own
    server path uses :func:`runner_pid_map` (one process listing + one
    manifest walk for all models).

    ``None`` if ``model_name`` is falsy, if Ollama isn't running that model,
    or if correlation fails for any reason (unexpected command-line shape,
    manifest missing/unparsable).
    """
    if not model_name:
        return None
    return runner_pid_map(
        list_processes=list_processes, manifests_root=manifests_root
    ).get(model_name)
