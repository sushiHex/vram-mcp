"""Tests for vram_mcp.ollama — pure, mocked session, no real Ollama."""

import requests
import pytest

from vram_mcp.ollama import OllamaClient


@pytest.mark.parametrize("payload", [{}, {"models": None}, {"models": [None]}, {"models": [{"name": " "}]}])
def test_invalid_observation_is_unknown_not_empty(payload):
    client = OllamaClient(session=FakeSession(get_response=FakeResponse(200, payload)))
    reading = client.observe_loaded()
    assert reading.data is None and not reading.known


def test_successful_empty_observation_is_known():
    client = OllamaClient(session=FakeSession(get_response=FakeResponse(200, {"models": []})))
    assert client.observe_loaded().known


def test_unload_timeout_reconciles_without_repeating_post():
    session = FakeSession(get_response=FakeResponse(200, {"models": []}), raise_on="post")
    result = OllamaClient(session=session).change_residency("llama3", 0, resident=False, sleep=lambda _: None)
    assert result["outcome"] == "succeeded"
    assert result["model"] == "llama3:latest"
    assert len(session.post_calls) == 1


def test_warm_timeout_does_not_claim_keep_alive_was_applied():
    session = FakeSession(get_response=FakeResponse(200, {"models": [{"name": "llama3:latest"}]}),
                          raise_on="post")
    result = OllamaClient(session=session).change_residency("llama3", "5m", resident=True, sleep=lambda _: None)
    assert result["outcome"] == "unknown"
    assert result["resident"] is True
    assert len(session.post_calls) == 1
    assert len(session.get_calls) == 3


def test_acknowledgement_requires_residency_verification():
    session = FakeSession(get_response=FakeResponse(200, {"models": []}), post_response=FakeResponse())
    result = OllamaClient(session=session).change_residency("llama3", "5m", resident=True, sleep=lambda _: None)
    assert result["outcome"] == "unknown"
    assert result["ok"] is False


def test_backend_rejection_is_definite_failure():
    session = FakeSession(post_response=FakeResponse(404))
    result = OllamaClient(session=session).change_residency("missing", "5m", resident=True)
    assert result["outcome"] == "failed"
    assert session.get_calls == []


def test_verified_warm_canonicalizes_alias_and_encodes_indefinite_duration():
    session = FakeSession(
        get_response=FakeResponse(200, {"models": [{"name": "Library/LLAMA3"}]}),
        post_response=FakeResponse(200, {"done": True}))
    result = OllamaClient(session=session).change_residency("llama3", "-1", resident=True)
    assert result["outcome"] == "succeeded"
    assert result["resident"] is True
    assert session.post_calls[0]["json"]["keep_alive"] == -1


def test_residency_changes_outwait_a_slow_load_but_connect_fast():
    """Ollama aborts a load when its client disconnects: 0.40 on Windows took
    over 10 s to load a 2.5 GB model, and the old read timeout killed it."""
    session = FakeSession(get_response=FakeResponse(200, {"models": []}),
                          post_response=FakeResponse(200, {"done": True}))
    client = OllamaClient(session=session)

    client.change_residency("llama3", 0, resident=False)
    client.warm("llama3")

    assert [call["timeout"] for call in session.post_calls] == [(10, 110)] * 2
    assert session.get_calls[0]["timeout"] == 10


def test_a_residency_request_ends_inside_its_operation_lease():
    from vram_mcp import claims
    assert OllamaClient().residency_timeout < claims._OPERATION_LEASE_SECONDS


def test_a_per_request_timeout_replaces_the_read_timeout_only():
    session = FakeSession(get_response=FakeResponse(200, {"models": [{"name": "llama3:latest"}]}),
                          post_response=FakeResponse(200, {"done": True}))
    result = OllamaClient(session=session).change_residency("llama3", "5m", resident=True, timeout=300)
    assert session.post_calls[0]["timeout"] == (10, 300)
    # The applied budget is reported, so a caller can see what it got.
    assert result["timeout_seconds"] == 300


def test_num_ctx_is_forwarded_as_an_ollama_option():
    session = FakeSession(get_response=FakeResponse(200, {"models": [{"name": "llama3:latest"}]}),
                          post_response=FakeResponse(200, {"done": True}))
    client = OllamaClient(session=session)
    client.change_residency("llama3", "5m", resident=True, num_ctx=65536)
    client.change_residency("llama3", "5m", resident=True)
    assert session.post_calls[0]["json"]["options"] == {"num_ctx": 65536}
    assert "options" not in session.post_calls[1]["json"]


def test_a_read_timeout_is_a_distinct_unknown_outcome():
    """A slow load used to read as a generic interruption; the caller could not
    tell "Ollama took longer than we waited" from a broken connection."""
    session = FakeSession(get_response=FakeResponse(200, {"models": []}),
                          raise_on="post", post_error=requests.ReadTimeout("read timed out"))
    result = OllamaClient(session=session).change_residency(
        "llama3", "5m", resident=True, timeout=110, sleep=lambda _: None)
    assert result["outcome"] == "unknown"
    assert result["reason"] == "load_timeout"
    assert result["timeout_seconds"] == 110
    assert "110 s" in result["detail"] and "ReadTimeout" in result["detail"]
    assert len(session.post_calls) == 1


def test_an_interrupted_request_names_its_exception():
    session = FakeSession(get_response=FakeResponse(200, {"models": []}), raise_on="post")
    result = OllamaClient(session=session).change_residency(
        "llama3", "5m", resident=True, sleep=lambda _: None)
    assert result["outcome"] == "unknown"
    assert result["reason"] == "request_interrupted"
    assert "ConnectionError" in result["detail"]


def test_observe_show_returns_model_info():
    info = {"general.architecture": "llama", "llama.block_count": 32}
    session = FakeSession(post_response=FakeResponse(200, {"model_info": info}))
    reading = OllamaClient(session=session).observe_show("llama3")
    assert reading.known and reading.data == info
    assert session.post_calls[0]["url"].endswith("/api/show")
    assert session.post_calls[0]["json"] == {"model": "llama3:latest"}
    assert session.post_calls[0]["timeout"] == 10


@pytest.mark.parametrize("make_session", [
    lambda: FakeSession(raise_on="post"),
    lambda: FakeSession(post_response=FakeResponse(404, {})),
    lambda: FakeSession(post_response=FakeResponse(200, {"model_info": None})),
])
def test_observe_show_failure_is_unknown_not_empty(make_session):
    reading = OllamaClient(session=make_session()).observe_show("llama3")
    assert not reading.known and reading.error


def test_invalid_acknowledgement_cannot_verify_keep_alive():
    session = FakeSession(
        get_response=FakeResponse(200, {"models": [{"name": "llama3:latest"}]}),
        post_response=FakeResponse(200, {"error": "bad request"}))
    result = OllamaClient(session=session).change_residency("llama3", "5m", resident=True, sleep=lambda _: None)
    assert result["outcome"] == "unknown"


class FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json = json_data if json_data is not None else {}

    def raise_for_status(self):
        if not (200 <= self.status_code < 300):
            raise requests.HTTPError(f"status {self.status_code}")

    def json(self):
        return self._json


class FakeSession:
    """Records GET/POST calls and returns queued responses."""

    def __init__(self, get_response=None, post_response=None, raise_on=None, post_error=None):
        self.get_response = get_response
        self.post_response = post_response
        self.raise_on = raise_on  # "get" | "post" | None
        self.post_error = post_error
        self.get_calls = []
        self.post_calls = []

    def get(self, url, timeout=None):
        self.get_calls.append({"url": url, "timeout": timeout})
        if self.raise_on == "get":
            raise requests.ConnectionError("refused")
        return self.get_response

    def post(self, url, json=None, timeout=None):
        self.post_calls.append({"url": url, "json": json, "timeout": timeout})
        if self.raise_on == "post":
            raise self.post_error or requests.ConnectionError("refused")
        return self.post_response


def test_ps_parses_models():
    models = [
        {"name": "llama3", "size": 5000000000, "size_vram": 4800000000,
         "expires_at": "2026-07-11T10:00:00Z"},
    ]
    session = FakeSession(get_response=FakeResponse(200, {"models": models}))
    client = OllamaClient(session=session)
    assert client.ps() == models
    assert session.get_calls[0]["url"] == "http://127.0.0.1:11434/api/ps"


def test_ps_missing_models_key_returns_empty():
    session = FakeSession(get_response=FakeResponse(200, {}))
    assert OllamaClient(session=session).ps() == []


def test_ps_transport_error_returns_empty():
    session = FakeSession(raise_on="get")
    assert OllamaClient(session=session).ps() == []


def test_ps_http_error_returns_empty():
    session = FakeSession(get_response=FakeResponse(500, {}))
    assert OllamaClient(session=session).ps() == []


def test_tags_returns_model_sizes():
    payload = {"models": [
        {"name": "llama3:8b", "size": 4 * 1024 * 1024 * 1024},
        {"name": "qwen3:32b", "size": 20 * 1024 * 1024 * 1024},
    ]}
    session = FakeSession(get_response=FakeResponse(200, payload))
    client = OllamaClient(session=session)
    tags = client.tags()
    assert tags["llama3:8b"] == 4096
    assert tags["qwen3:32b"] == 20480
    assert session.get_calls[0]["url"] == "http://127.0.0.1:11434/api/tags"


def test_tags_transport_error_returns_empty():
    session = FakeSession(raise_on="get")
    assert OllamaClient(session=session).tags() == {}


def test_tags_http_error_returns_empty():
    session = FakeSession(get_response=FakeResponse(500, {}))
    assert OllamaClient(session=session).tags() == {}


def test_tags_missing_models_key_returns_empty():
    session = FakeSession(get_response=FakeResponse(200, {}))
    assert OllamaClient(session=session).tags() == {}


def test_tags_skips_nameless_rows():
    payload = {"models": [{"size": 1024}, {"name": "a", "size": 1048576}]}
    session = FakeSession(get_response=FakeResponse(200, payload))
    assert OllamaClient(session=session).tags() == {"a": 1}


def test_tags_skips_unknown_sizes_and_non_dict_rows():
    payload = {"models": [
        "not-a-dict",
        {"name": "nosize"},
        {"name": "garbage", "size": "big"},
        {"name": "ok", "size": 2 * 1024 * 1024},
    ]}
    session = FakeSession(get_response=FakeResponse(200, payload))
    assert OllamaClient(session=session).tags() == {"ok": 2}


def test_unload_posts_keep_alive_zero_and_returns_true():
    session = FakeSession(post_response=FakeResponse(200))
    client = OllamaClient(session=session)
    assert client.unload("llama3") is True

    call = session.post_calls[0]
    assert call["url"] == "http://127.0.0.1:11434/api/generate"
    assert call["json"]["model"] == "llama3"
    assert call["json"]["keep_alive"] == 0
    assert call["json"]["prompt"] == ""
    assert call["json"]["stream"] is False


def test_warm_posts_keep_alive_duration_and_returns_true():
    session = FakeSession(post_response=FakeResponse(200))
    client = OllamaClient(session=session)
    assert client.warm("llama3") is True
    assert session.post_calls[0]["json"]["keep_alive"] == "5m"


def test_warm_custom_keep_alive():
    session = FakeSession(post_response=FakeResponse(200))
    client = OllamaClient(session=session)
    assert client.warm("llama3", keep_alive="1h") is True
    assert session.post_calls[0]["json"]["keep_alive"] == "1h"


def test_unload_non_2xx_returns_false():
    session = FakeSession(post_response=FakeResponse(404))
    assert OllamaClient(session=session).unload("nope") is False


def test_warm_transport_error_returns_false():
    session = FakeSession(raise_on="post")
    assert OllamaClient(session=session).warm("llama3") is False


def test_base_url_trailing_slash_stripped():
    session = FakeSession(post_response=FakeResponse(200))
    client = OllamaClient(base_url="http://host:11434/", session=session)
    client.unload("m")
    assert session.post_calls[0]["url"] == "http://host:11434/api/generate"
