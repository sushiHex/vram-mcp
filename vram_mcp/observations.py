"""A reading carries its provenance; an unavailable source never means empty."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Observation(Generic[T]):
    data: T | None
    source: str
    observed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    error: str | None = None
    scope: str | None = None
    coverage: dict[str, bool] | None = None
    # Why each uncovered input is missing, keyed like ``coverage``.
    gaps: dict[str, str] | None = None

    @property
    def known(self) -> bool:
        return self.data is not None and self.error is None

    def metadata(self) -> dict:
        result = {"status": "available" if self.known else "unavailable",
                  "source": self.source, "observed_at": self.observed_at,
                  "scope": self.scope}
        if self.error is not None:
            result["error"] = self.error
        if self.coverage is not None:
            result["coverage"] = self.coverage
        if self.gaps:
            result["gaps"] = self.gaps
        return result
