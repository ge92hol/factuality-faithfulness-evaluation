"""Common adapter contract (spec §7). Native metrics only: no universal 0-1 score."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

EXTERNAL = Path(__file__).resolve().parents[3] / "external"

OK, NOT_APPLICABLE, NOT_RUN, ERROR = "OK", "NOT_APPLICABLE", "NOT_RUN", "ERROR"

# Recorded in every run's run_meta.json and in outputs/results/results_meta.json.
IMPLEMENTATION_VERSION = "1.0.0"


def git_sha(repo: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(EXTERNAL / repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


@dataclasses.dataclass
class MetricValue:
    evaluator_name: str
    mode: str                      # evaluator-specific sub-protocol (e.g. published_cs)
    native_metric_name: str
    native_metric_value: Optional[float]
    higher_is_better: Optional[bool]
    native_range: str
    status: str = OK
    warnings: List[str] = dataclasses.field(default_factory=list)
    reference_dependent: Optional[bool] = None  # True iff the value depends on a reference answer


@dataclasses.dataclass
class EvaluationResult:
    evaluator_name: str
    mode: str
    implementation: str            # package/version or repo@sha
    adaptation: List[str]          # every deviation from the published configuration
    judge: Dict[str, Any]
    status: str
    metrics: List[MetricValue]
    raw: Dict[str, Any]            # full native output (claims, per-doc labels, ...)
    metadata_consumed: Dict[str, bool]
    runtime_sec: float = 0.0
    judge_usage: Dict[str, Any] = dataclasses.field(default_factory=dict)
    warnings: List[str] = dataclasses.field(default_factory=list)
    error: Optional[str] = None
    config_hash: str = ""
    implementation_version: str = IMPLEMENTATION_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def config_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


# Which scenario metadata a published implementation actually ingests. Verified
# by reading each implementation's input interface (see README §Adapters).
TEXT_ONLY_METADATA = {"reliability": False, "source_hedge_label": False, "source_id": False,
                      "doc_order": True, "doc_text": True}


class EvaluatorAdapter:
    name = "abstract"
    modes: List[str] = []

    def __init__(self, judge) -> None:
        self.judge = judge

    def evaluate(self, query: str, answer: str, contexts: List[str], scenario_metadata: Dict,
                 reference: Optional[str] = None, mode: Optional[str] = None) -> EvaluationResult:
        raise NotImplementedError
