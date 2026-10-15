"""Ollama transport with explicit read health and verified mutation outcomes."""

from __future__ import annotations

import time
from typing import Optional

import requests

from ._util import bytes_to_mb
from .observations import Observation
from .models import canonical_model


class OllamaClient:
    """Thin client over the Ollama REST API used for VRAM management.

    Only the endpoints needed to see, size and evict resident models are
    wired up: ``/api/ps`` (list loaded), ``/api/tags`` (on-disk sizes), and
    ``/api/generate`` with a ``keep_alive`` of ``0`` (unload) or a duration
    (warm/pin).
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        session: Optional[requests.Session] = None,
        timeout: int = 10,
        residency_timeout: int = 110,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.timeout = timeout
        # Ollama cancels a load when its client disconnects, so a request that
        # changes residency must outwait a slow load rather than abort it. It
        # still ends inside the 120 s operation lease, whose pending window
        # then covers anything the server finishes afterwards. Connecting keeps
        # the short timeout: an unreachable Ollama should fail fast.
        self.residency_timeout = residency_timeout

    def _post_generate(self, payload: dict):
        return self.session.post(self._url("/api/generate"), json=payload,
                                 timeout=(self.timeout, self.residency_timeout))

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _get_json(self, path: str):
        """GET ``path`` and return the decoded JSON, or ``None`` on ANY failure.

        One place for the read-side transport and failure policy, so every
        GET endpoint degrades identically (connection refused, timeout, non-2xx
        and undecodable body all collapse to ``None``) instead of each method
        inventing its own idea of "Ollama is down".
        """
        try:
            resp = self.session.get(self._url(path), timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError):
            return None

    def ps(self) -> list[dict]:
        """Return the list of currently loaded models (``/api/ps``).

        Each model has at least ``name``, ``size`` (bytes), ``size_vram``
        (bytes) and ``expires_at``. Returns ``[]`` on any transport error.
        """
        data = self._get_json("/api/ps")
        if not isinstance(data, dict):
            return []
        models = data.get("models")
        return models if isinstance(models, list) else []

    def observe_loaded(self) -> Observation[list[dict]]:
        payload = self._get_json("/api/ps")
        rows = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(rows, list) or any(
            not isinstance(row, dict) or not isinstance(row.get("name"), str)
            or not row["name"].strip() for row in rows
        ):
            return Observation(None, "ollama:/api/ps", error="Ollama unavailable or invalid response",
                               scope=self.base_url)
        try:
            rows = [{**row, "name": canonical_model(row["name"])} for row in rows]
        except ValueError:
            return Observation(None, "ollama:/api/ps", error="Invalid model name", scope=self.base_url)
        return Observation(rows, "ollama:/api/ps", scope=self.base_url)

    def observe_tags(self) -> Observation[dict]:
        payload = self._get_json("/api/tags")
        rows = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            return Observation(None, "ollama:/api/tags", error="Ollama unavailable or invalid response",
                               scope=self.base_url)
        sizes = {}
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("name"), str) or not row["name"].strip():
                return Observation(None, "ollama:/api/tags", error="Invalid model entry", scope=self.base_url)
            size = bytes_to_mb(row.get("size"))
            if size is not None and size > 0:
                try:
                    sizes[canonical_model(row["name"])] = size
                except ValueError:
                    return Observation(None, "ollama:/api/tags", error="Invalid model name", scope=self.base_url)
        return Observation(sizes, "ollama:/api/tags", scope=self.base_url)

    def change_residency(self, model: str, keep_alive, *, resident: bool,
                         attempts: int = 3, sleep=time.sleep) -> dict:
        """Send once; reconcile with bounded reads, never blindly retry a POST.

        A timeout may have changed the server. For warm, residency alone cannot
        prove that the requested keep-alive was applied, so no acknowledgement
        remains an unknown outcome even if the model is now resident.
        """
        model = canonical_model(model)
        acknowledged = False
        error = None
        try:
            response = self._post_generate({
                "model": model, "prompt": "", "keep_alive": -1 if keep_alive == "-1" else keep_alive,
                "stream": False,
            })
            acknowledged = 200 <= response.status_code < 300
            if 400 <= response.status_code < 500:
                return {"ok": False, "outcome": "failed", "model": model,
                        "detail": f"Ollama rejected the request (HTTP {response.status_code})"}
            if not acknowledged:
                error = f"Ollama returned HTTP {response.status_code}"
            else:
                try:
                    payload = response.json()
                except ValueError:
                    payload = None
                acknowledged = isinstance(payload, dict) and payload.get("done") is True and not payload.get("error")
                if not acknowledged:
                    error = "Ollama did not return a valid completion acknowledgement"
        except requests.RequestException:
            error = "Ollama request was interrupted; it may still be running"
        reading = None
        observed_resident = None
        for attempt in range(attempts):
            if attempt:
                sleep(0.2)
            reading = self.observe_loaded()
            if reading.known:
                observed_resident = any(canonical_model(row["name"]) == model for row in reading.data)
                if observed_resident == resident and (acknowledged or not resident):
                    return {"ok": True, "outcome": "succeeded", "model": model,
                            "resident": observed_resident, "observations": {"ollama": reading.metadata()},
                            "detail": "Residency verified after request"}
        return {"ok": False, "outcome": "unknown", "model": model,
                "resident": observed_resident,
                "observations": {"ollama": reading.metadata()} if reading else {},
                "detail": error or "Ollama acknowledged the request, but residency could not be verified"}

    def tags(self) -> dict:
        """``{model_name: size_mb}`` from ``/api/tags`` (on-disk sizes).

        Disk size approximates VRAM need — close enough to decide whether a
        warm plausibly fits, never precise enough to be authoritative. ``{}``
        on any failure, so callers must treat a missing entry as "unknown"
        rather than "zero".
        """
        payload = self._get_json("/api/tags")
        if not isinstance(payload, dict):
            return {}
        sizes: dict = {}
        for row in payload.get("models") or []:
            if not isinstance(row, dict):
                continue
            name = row.get("name")
            if not name:
                continue
            size_mb = bytes_to_mb(row.get("size"), default=None)
            if size_mb is None:
                continue
            sizes[name] = size_mb
        return sizes

    def _generate(self, model: str, keep_alive) -> bool:
        """POST an empty generate with a ``keep_alive`` to load/unload a model."""
        payload = {
            "model": model,
            "prompt": "",
            "keep_alive": keep_alive,
            "stream": False,
        }
        try:
            resp = self._post_generate(payload)
        except requests.RequestException:
            return False
        return 200 <= resp.status_code < 300

    def unload(self, model: str) -> bool:
        """Evict ``model`` from VRAM now (``keep_alive=0``). True on 2xx."""
        return self._generate(model, 0)

    def warm(self, model: str, keep_alive: str = "5m") -> bool:
        """Load/pin ``model`` for ``keep_alive`` (e.g. ``"5m"``). True on 2xx."""
        return self._generate(model, keep_alive)
