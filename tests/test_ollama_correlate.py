"""Tests for vram_mcp.ollama_correlate — pure except real tmp_path manifests."""

import json

from vram_mcp import ollama_correlate as oc

_REAL_DIGEST = "a3de86cd1c132c822487ededd47a324c50491393e6565cd14bafa40d0b8e686f"
_MMPROJ_DIGEST = "c" * 64


def _write_manifest(manifests_root, registry_host, namespace, name, tag, model_digest):
    """Write a manifest at the REAL Ollama on-disk depth:
    <manifests_root>/<registry_host>/<namespace>/<name>/<tag>
    (verified live: registry.ollama.ai/library/gemma3/27b and
    hf.co/NousResearch/Hermes-4.3-36B-GGUF/q4_K_M)."""
    path = manifests_root / registry_host / namespace / name / tag
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schemaVersion": 2,
        "layers": [
            {"mediaType": "application/vnd.ollama.image.model",
             "digest": f"sha256:{model_digest}"},
            {"mediaType": "application/vnd.ollama.image.template",
             "digest": "sha256:deadbeef"},
        ],
    }))


# ---------------------------------------------------------------- digests


def test_extract_blob_digests_from_real_cmdline():
    cmdline = (
        r"C:\Ollama\llama-server.exe --model "
        rf"C:\Users\x\.ollama\models\blobs\sha256-{_REAL_DIGEST} --port 12288"
    )
    assert oc._extract_blob_digests(cmdline) == [_REAL_DIGEST]


def test_extract_blob_digests_quoted_spaced_path():
    # A username with a space plus shell quoting must not truncate the digest.
    cmdline = (
        r'"C:\Program Files\Ollama\llama-server.exe" --model '
        rf'"C:\Users\John Smith\.ollama\models\blobs\sha256-{_REAL_DIGEST}" --port 1'
    )
    assert oc._extract_blob_digests(cmdline) == [_REAL_DIGEST]


def test_extract_blob_digests_multiple_blobs_multimodal():
    # --model plus --mmproj (multimodal runner) → both digests returned, in order.
    cmdline = (
        rf"llama-server.exe --model C:\x\blobs\sha256-{_REAL_DIGEST} "
        rf"--mmproj C:\x\blobs\sha256-{_MMPROJ_DIGEST} --port 1"
    )
    assert oc._extract_blob_digests(cmdline) == [_REAL_DIGEST, _MMPROJ_DIGEST]


def test_extract_blob_digests_lowercases_and_dedupes():
    upper = _REAL_DIGEST.upper()
    cmdline = rf"exe --model /a/sha256-{upper} --draft /b/sha256-{_REAL_DIGEST}"
    assert oc._extract_blob_digests(cmdline) == [_REAL_DIGEST]


def test_extract_blob_digests_none_found():
    assert oc._extract_blob_digests("llama-server.exe --port 12288") == []


def test_extract_blob_digests_short_hash_not_matched():
    # A malformed/truncated digest (not 64 hex chars) is never extracted —
    # never guess a partial match.
    assert oc._extract_blob_digests("llama-server.exe --model /x/sha256-abc123") == []


# ---------------------------------------------------------- digest → tags


def test_digest_to_tags_library_namespace(tmp_path):
    manifests = tmp_path / "manifests"
    _write_manifest(manifests, "registry.ollama.ai", "library", "qwen3", "8b", "abc123")
    assert oc._digest_to_tags(manifests) == {"abc123": {"qwen3:8b"}}


def test_digest_to_tags_non_library_namespace(tmp_path):
    manifests = tmp_path / "manifests"
    _write_manifest(manifests, "registry.ollama.ai", "someuser", "mymodel", "latest", "abc123")
    assert oc._digest_to_tags(manifests) == {"abc123": {"someuser/mymodel:latest"}}


def test_digest_to_tags_hf_co_host_preserved(tmp_path):
    # Any non-official registry host keeps the host in the tag name,
    # matching Ollama's /api/ps naming (verified against real hf.co manifests).
    manifests = tmp_path / "manifests"
    _write_manifest(manifests, "hf.co", "NousResearch",
                    "Hermes-4.3-36B-GGUF", "q4_K_M", "abc123")
    assert oc._digest_to_tags(manifests) == {
        "abc123": {"hf.co/NousResearch/Hermes-4.3-36B-GGUF:q4_K_M"},
    }


def test_digest_to_tags_shared_digest_multimap(tmp_path):
    # One blob shared by several tags (ollama cp / re-tags / hf.co variants):
    # ALL tag names must appear under the same digest.
    manifests = tmp_path / "manifests"
    _write_manifest(manifests, "registry.ollama.ai", "library",
                    "hermes-4.3-36b-q4km-32k", "latest", _REAL_DIGEST)
    _write_manifest(manifests, "hf.co", "NousResearch",
                    "Hermes-4.3-36B-GGUF", "q4_K_M", _REAL_DIGEST)
    _write_manifest(manifests, "registry.ollama.ai", "library",
                    "qwen3", "8b", "other000")
    assert oc._digest_to_tags(manifests) == {
        _REAL_DIGEST: {"hermes-4.3-36b-q4km-32k:latest",
                       "hf.co/NousResearch/Hermes-4.3-36B-GGUF:q4_K_M"},
        "other000": {"qwen3:8b"},
    }


def test_digest_to_tags_missing_dir_returns_empty(tmp_path):
    assert oc._digest_to_tags(tmp_path / "nope") == {}


def test_digest_to_tags_wrong_depth_skipped(tmp_path):
    # A manifest-shaped file at the WRONG depth (3 levels, missing the
    # registry-host segment) is skipped, not guessed at.
    manifests = tmp_path / "manifests"
    path = manifests / "library" / "qwen3" / "8b"  # only 3 levels deep
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "layers": [{"mediaType": "application/vnd.ollama.image.model",
                    "digest": "sha256:abc123"}],
    }))
    assert oc._digest_to_tags(manifests) == {}


def test_digest_to_tags_unparsable_manifest_skipped(tmp_path):
    manifests = tmp_path / "manifests"
    _write_manifest(manifests, "registry.ollama.ai", "library", "qwen3", "8b", "abc123")
    bad = manifests / "registry.ollama.ai" / "library" / "broken" / "latest"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("not json {")
    assert oc._digest_to_tags(manifests) == {"abc123": {"qwen3:8b"}}


# ------------------------------------------------------------ pid mapping


def test_runner_pid_map_end_to_end_shared_digest(tmp_path):
    # One runner serving a digest shared by 2 tags → BOTH tags map to the pid.
    manifests = tmp_path / "manifests"
    _write_manifest(manifests, "registry.ollama.ai", "library",
                    "hermes-4.3-36b-q4km-32k", "latest", _REAL_DIGEST)
    _write_manifest(manifests, "hf.co", "NousResearch",
                    "Hermes-4.3-36B-GGUF", "q4_K_M", _REAL_DIGEST)

    def fake_list_processes():
        return [{"pid": 43176, "cmdline":
                 rf"llama-server.exe --model C:\x\blobs\sha256-{_REAL_DIGEST} --port 1"}]

    assert oc.runner_pid_map(
        list_processes=fake_list_processes, manifests_root=manifests
    ) == {
        "hermes-4.3-36b-q4km-32k:latest": 43176,
        "hf.co/NousResearch/Hermes-4.3-36B-GGUF:q4_K_M": 43176,
    }


def test_runner_pid_map_multiple_runners(tmp_path):
    manifests = tmp_path / "manifests"
    digest_a, digest_b = "a" * 64, "b" * 64
    _write_manifest(manifests, "registry.ollama.ai", "library", "qwen3", "8b", digest_a)
    _write_manifest(manifests, "registry.ollama.ai", "library", "gemma3", "27b", digest_b)

    def fake_list_processes():
        return [
            {"pid": 111, "cmdline": rf"llama-server.exe --model /b/sha256-{digest_a}"},
            {"pid": 222, "cmdline": rf"llama-server.exe --model /b/sha256-{digest_b}"},
        ]

    assert oc.runner_pid_map(
        list_processes=fake_list_processes, manifests_root=manifests
    ) == {"qwen3:8b": 111, "gemma3:27b": 222}


def test_runner_pid_map_mmproj_digest_ignored(tmp_path):
    # A projector blob digest has no model-layer manifest entry, so it
    # never matches — only the model tag appears.
    manifests = tmp_path / "manifests"
    _write_manifest(manifests, "registry.ollama.ai", "library", "gemma3", "27b", _REAL_DIGEST)

    def fake_list_processes():
        return [{"pid": 7, "cmdline": (
            rf"llama-server.exe --model C:\x\blobs\sha256-{_REAL_DIGEST} "
            rf"--mmproj C:\x\blobs\sha256-{_MMPROJ_DIGEST}")}]

    assert oc.runner_pid_map(
        list_processes=fake_list_processes, manifests_root=manifests
    ) == {"gemma3:27b": 7}


def test_runner_pid_map_no_processes_returns_empty(tmp_path):
    manifests = tmp_path / "manifests"
    _write_manifest(manifests, "registry.ollama.ai", "library", "qwen3", "8b", _REAL_DIGEST)
    assert oc.runner_pid_map(list_processes=lambda: [], manifests_root=manifests) == {}


def test_runner_pid_map_missing_manifests_returns_empty(tmp_path):
    def fake_list_processes():
        return [{"pid": 1, "cmdline": rf"exe --model /b/sha256-{_REAL_DIGEST}"}]

    assert oc.runner_pid_map(
        list_processes=fake_list_processes, manifests_root=tmp_path / "nope"
    ) == {}


def test_runner_pid_map_list_processes_raises_returns_empty(tmp_path):
    def boom():
        raise RuntimeError("ps exploded")

    assert oc.runner_pid_map(list_processes=boom, manifests_root=tmp_path) == {}


def test_runner_pid_map_malformed_process_entries_skipped(tmp_path):
    manifests = tmp_path / "manifests"
    _write_manifest(manifests, "registry.ollama.ai", "library", "qwen3", "8b", _REAL_DIGEST)

    def fake_list_processes():
        return [
            {"pid": "not-an-int", "cmdline": rf"exe --model /b/sha256-{_REAL_DIGEST}"},
            {"pid": 5},  # missing cmdline
            {"pid": 9, "cmdline": rf"exe --model /b/sha256-{_REAL_DIGEST}"},
        ]

    assert oc.runner_pid_map(
        list_processes=fake_list_processes, manifests_root=manifests
    ) == {"qwen3:8b": 9}


# ------------------------------------------------------ find_pid_for_model


def test_find_pid_for_model_matches_correlated_tag(tmp_path):
    manifests = tmp_path / "manifests"
    _write_manifest(manifests, "registry.ollama.ai", "library", "qwen3", "8b", _REAL_DIGEST)

    def fake_list_processes():
        return [{"pid": 43176, "cmdline":
                 rf"llama-server.exe --model C:\x\blobs\sha256-{_REAL_DIGEST} --port 1"}]

    pid = oc.find_pid_for_model(
        "qwen3:8b", list_processes=fake_list_processes, manifests_root=manifests
    )
    assert pid == 43176


def test_find_pid_for_model_spaced_quoted_path(tmp_path):
    # The original --model\s+(\S+) regex truncated at the first space in a
    # quoted path; digest-anywhere extraction must survive it.
    manifests = tmp_path / "manifests"
    _write_manifest(manifests, "registry.ollama.ai", "library", "qwen3", "8b", _REAL_DIGEST)

    def fake_list_processes():
        return [{"pid": 314, "cmdline": (
            r'"C:\Program Files\Ollama\llama-server.exe" --model '
            rf'"C:\Users\John Smith\.ollama\models\blobs\sha256-{_REAL_DIGEST}" --port 1')}]

    pid = oc.find_pid_for_model(
        "qwen3:8b", list_processes=fake_list_processes, manifests_root=manifests
    )
    assert pid == 314


def test_find_pid_for_model_every_alias_matches(tmp_path):
    manifests = tmp_path / "manifests"
    _write_manifest(manifests, "registry.ollama.ai", "library",
                    "hermes-4.3-36b-q4km-32k", "latest", _REAL_DIGEST)
    _write_manifest(manifests, "hf.co", "NousResearch",
                    "Hermes-4.3-36B-GGUF", "q4_K_M", _REAL_DIGEST)

    def fake_list_processes():
        return [{"pid": 999, "cmdline":
                 rf"llama-server.exe --model /b/sha256-{_REAL_DIGEST}"}]

    for alias in ("hermes-4.3-36b-q4km-32k:latest",
                  "hf.co/NousResearch/Hermes-4.3-36B-GGUF:q4_K_M"):
        assert oc.find_pid_for_model(
            alias, list_processes=fake_list_processes, manifests_root=manifests
        ) == 999


def test_find_pid_for_model_none_name_returns_none(tmp_path):
    # Guard: a falsy model name never false-matches, even when a runner's
    # digest fails to resolve to any tag.
    manifests = tmp_path / "manifests"  # empty → digest resolves to nothing

    def fake_list_processes():
        return [{"pid": 1, "cmdline": rf"exe --model /b/sha256-{_REAL_DIGEST}"}]

    assert oc.find_pid_for_model(
        None, list_processes=fake_list_processes, manifests_root=manifests
    ) is None
    assert oc.find_pid_for_model(
        "", list_processes=fake_list_processes, manifests_root=manifests
    ) is None


def test_find_pid_for_model_no_matching_process_returns_none(tmp_path):
    manifests = tmp_path / "manifests"
    _write_manifest(manifests, "registry.ollama.ai", "library", "qwen3", "8b", _REAL_DIGEST)
    pid = oc.find_pid_for_model(
        "qwen3:8b", list_processes=lambda: [], manifests_root=manifests
    )
    assert pid is None


def test_find_pid_for_model_digest_mismatch_returns_none(tmp_path):
    manifests = tmp_path / "manifests"
    _write_manifest(manifests, "registry.ollama.ai", "library", "qwen3", "8b", _REAL_DIGEST)
    other_digest = "b" * 64

    def fake_list_processes():
        return [{"pid": 1, "cmdline":
                 rf"llama-server.exe --model C:\x\blobs\sha256-{other_digest}"}]

    pid = oc.find_pid_for_model(
        "qwen3:8b", list_processes=fake_list_processes, manifests_root=manifests
    )
    assert pid is None


# ------------------------------------------------- windows process listing


def _fake_winproc(monkeypatch, names, cmdlines):
    monkeypatch.setattr(oc.winproc, "process_names", lambda: dict(names))
    monkeypatch.setattr(oc.winproc, "command_line", lambda pid: cmdlines.get(pid))


def test_windows_lists_llama_server_processes_with_cmdlines(monkeypatch):
    cmdline = rf"C:\Ollama\llama-server.exe --model C:\x\blobs\sha256-{_REAL_DIGEST}"
    _fake_winproc(monkeypatch,
                  {7: "llama-server.exe", 8: "ollama.exe", 9: "notepad.exe"},
                  {7: cmdline, 8: "ollama.exe serve"})
    assert oc._list_llama_server_processes_windows() == [
        {"pid": 7, "cmdline": cmdline}]


def test_windows_llama_server_name_matches_case_insensitively(monkeypatch):
    _fake_winproc(monkeypatch, {7: "LLAMA-Server.EXE"}, {7: "runner"})
    assert oc._list_llama_server_processes_windows() == [
        {"pid": 7, "cmdline": "runner"}]


def test_windows_runner_without_a_command_line_is_excluded(monkeypatch):
    _fake_winproc(monkeypatch, {7: "llama-server.exe", 8: "llama-server.exe"},
                  {8: "runner"})
    assert oc._list_llama_server_processes_windows() == [
        {"pid": 8, "cmdline": "runner"}]


def test_windows_listing_degrades_to_empty_on_oserror(monkeypatch):
    def fail():
        raise OSError("Windows process APIs are unavailable")

    monkeypatch.setattr(oc.winproc, "process_names", fail)
    assert oc._list_llama_server_processes_windows() == []


def test_list_llama_server_processes_dispatches_windows_without_timeout(monkeypatch):
    monkeypatch.setattr(oc.sys, "platform", "win32")
    monkeypatch.setattr(oc, "_list_llama_server_processes_windows",
                        lambda: [{"pid": 1, "cmdline": "x"}])
    assert oc._list_llama_server_processes() == [{"pid": 1, "cmdline": "x"}]


# ---- resolve_tag: Ollama's bare-name -> ":latest" rule ----------------------

def test_resolve_tag_exact_match_wins():
    known = {"qwen3:32b": 19265, "llama3.2:latest": 1900}
    assert oc.resolve_tag("qwen3:32b", known) == "qwen3:32b"


def test_resolve_tag_bare_name_resolves_to_latest():
    # Ollama itself resolves a bare name to its ":latest" tag, so a caller
    # asking for "llama3.2" gets a real multi-GB model — its size must be
    # findable, or the admission check silently degrades to "size unknown".
    known = {"llama3.2:latest": 1900}
    assert oc.resolve_tag("llama3.2", known) == "llama3.2:latest"


def test_resolve_tag_unknown_name_is_none():
    known = {"llama3.2:latest": 1900}
    assert oc.resolve_tag("nope", known) is None
    assert oc.resolve_tag("nope:7b", known) is None
    assert oc.resolve_tag("", known) is None
    assert oc.resolve_tag(None, known) is None


def test_resolve_tag_tagged_name_never_falls_back_to_latest():
    # "qwen3:32b" must NOT be answered with "qwen3:latest" — a different model
    # of a different size; a wrong size is worse than an unknown one.
    known = {"qwen3:latest": 5000}
    assert oc.resolve_tag("qwen3:32b", known) is None


def test_resolve_tag_registry_host_with_a_port_is_not_a_tag():
    # A colon in the REGISTRY HOST is not a tag: only the part after the last
    # "/" can carry one (same rule _tag_name_from_manifest_parts builds names
    # by). Reading "localhost:5000/library/foo" as already-tagged skipped the
    # :latest fallback and reported the model's size as unknown.
    known = {"localhost:5000/library/foo:latest": 4096}
    assert oc.resolve_tag("localhost:5000/library/foo",
                          known) == "localhost:5000/library/foo:latest"


def test_resolve_tag_non_official_registry_name_resolves_to_latest():
    known = {"hf.co/NousResearch/Hermes-4.3-36B-GGUF:latest": 21000}
    assert oc.resolve_tag(
        "hf.co/NousResearch/Hermes-4.3-36B-GGUF",
        known) == "hf.co/NousResearch/Hermes-4.3-36B-GGUF:latest"


def test_resolve_tag_tagged_name_under_a_ported_host_stays_tagged():
    known = {"localhost:5000/library/foo:latest": 4096}
    assert oc.resolve_tag("localhost:5000/library/foo:q4", known) is None
