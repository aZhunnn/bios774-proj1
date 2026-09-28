"""Runtime artifacts produced by compositional analysis steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd


@dataclass
class AnalysisArtifact:
    artifact_id: str
    data: Any
    role: Literal["root", "intermediate", "terminal"]
    source_step_id: str | None = None
    method: str | None = None
    parent_ref: str | None = None
    sample_ids: pd.Index | None = None
    signature: str | None = None
    n_features: int | None = None
