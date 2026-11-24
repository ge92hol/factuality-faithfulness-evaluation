"""RAGChecker adapter (amazon-science/RAGChecker 0.1.9 + RefChecker 0.2.17, Apache-2.0).

Uses the official `RAGChecker.evaluate` unchanged. The LLM is supplied through the
package's own `custom_llm_api_func` hook (documented constructor argument).

Reference modes (spec §7): RAGChecker requires `gt_answer` for every metric except
`faithfulness`. A conflicting document pair has no single reliable ground truth, so
each run is labelled with the reference it used and none is treated as truth:
  REF_A, REF_B, SHARED_MONTH, CONFLICT_AWARE  -> all 11 native metrics
  NO_REFERENCE                                -> faithfulness only (others NOT_APPLICABLE)

ADAPTATIONS: judge substituted (paper: Llama3-70B-Instruct as extractor and checker; code
default bedrock/meta.llama3-70b-instruct-v1:0; README examples use llama3-1-70b);
temperature=0 not enforceable via judge; reference answers are experimental settings,
not gold labels.
"""

from __future__ import annotations

import contextlib
import io
import threading
import time
from typing import Dict, List, Optional

from ..data_factory import REFERENCES
from ..judges.base import JudgeError
from .base import (ERROR, NOT_APPLICABLE, OK, TEXT_ONLY_METADATA, EvaluationResult, EvaluatorAdapter,
                   MetricValue, config_hash)

METRICS = ["precision", "recall", "f1", "claim_recall", "context_precision", "context_utilization",
           "noise_sensitivity_in_relevant", "noise_sensitivity_in_irrelevant", "hallucination",
           "self_knowledge", "faithfulness"]
HIGHER_BETTER = {"noise_sensitivity_in_relevant": False, "noise_sensitivity_in_irrelevant": False,
                 "hallucination": False, "self_knowledge": None}
REF_MODES = ["GOLDEN", "REF_A", "REF_B", "SHARED_MONTH", "CONFLICT_AWARE", "NO_REFERENCE"]
# GOLDEN: each item supplies item["reference"] = {"reference_id", "text", ...} (frozen per scenario x question)
REFERENCE_DEPENDENT = {m: True for m in ["precision", "recall", "f1", "claim_recall", "context_precision",
                                         "context_utilization", "noise_sensitivity_in_relevant",
                                         "noise_sensitivity_in_irrelevant", "hallucination", "self_knowledge"]}
REFERENCE_DEPENDENT["faithfulness"] = False


class RAGCheckerAdapter(EvaluatorAdapter):
    name = "RAGChecker"
    modes = REF_MODES

    def __init__(self, judge, joint_check: bool = True):
        super().__init__(judge)
        import ragchecker, refchecker  # noqa: F401
        from importlib.metadata import version
        self.impl = f"ragchecker=={version('ragchecker')}, refchecker=={version('refchecker')}"
        self.joint_check = joint_check
        self._parse_warn = threading.local()

    def _llm(self, prompts):
        out = []
        for p in prompts:
            if not isinstance(p, str):
                raise TypeError(f"unexpected prompt type {type(p)}")
        rs = self.judge.batch(list(prompts), tag="ragchecker")  # raises JudgeError on failure
        return [r.text for r in rs]

    def _make(self):
        from ragchecker import RAGChecker
        ev = RAGChecker(extractor_name="custom/judge", checker_name="custom/judge",
                        custom_llm_api_func=self._llm, joint_check=self.joint_check,
                        batch_size_extractor=32, batch_size_checker=32)
        # Observe (not alter) the official joint-label parser: RefChecker pads missing
        # labels with "Neutral" silently; we count responses that parsed no label.
        orig = ev.checker._parse_joint_checking_labels
        warns = []

        def wrapped(text):
            labels = orig(text)
            if not labels:
                warns.append(f"joint-check response parsed 0 labels (padded Neutral by RefChecker): {text[:120]!r}")
            return labels
        ev.checker._parse_joint_checking_labels = wrapped
        return ev, warns

    def evaluate(self, query, answer, contexts, scenario_metadata, reference=None, mode="SHARED_MONTH"):
        return self.evaluate_many([dict(item_id="single", query=query, answer=answer, contexts=contexts,
                                        scenario_metadata=scenario_metadata)], mode=mode)[0]

    def evaluate_many(self, items: List[Dict], mode: str = "SHARED_MONTH") -> List[EvaluationResult]:
        from ragchecker import RAGResults
        from ragchecker.metrics import all_metrics, faithfulness
        t0 = time.time()
        def gt_of(it):
            if mode == "GOLDEN":
                return it["reference"]["text"]
            return REFERENCES.get(mode, "") if mode != "NO_REFERENCE" else ""
        gts = [gt_of(it) for it in items]
        payload = {"results": [{"query_id": it["item_id"], "query": it["query"], "gt_answer": gts[i],
                                "response": it["answer"],
                                "retrieved_context": [{"doc_id": f"ctx{j}", "text": c} for j, c in enumerate(it["contexts"])]}
                               for i, it in enumerate(items)]}
        rr = RAGResults.from_dict(payload)
        ev, warns = self._make()
        led = self.judge.open_ledger()
        err = None
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                ev.evaluate(rr, faithfulness if mode == "NO_REFERENCE" else all_metrics)
        except (JudgeError, Exception) as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        finally:
            self.judge.close_ledger(led)
        rt = (time.time() - t0) / max(1, len(items))
        adapt = [f"judge substituted: {self.judge.describe()} (paper: Llama3-70B-Instruct extractor+checker; "
                 "code default bedrock/meta.llama3-70b-instruct-v1:0; README examples meta.llama3-1-70b-instruct-v1:0)",
                 "temperature=0 requested by official code but not enforceable via judge backend",
                 ]
        out = []
        for it, r, gt in zip(items, rr.results, gts):
            ref_id = it["reference"]["reference_id"] if mode == "GOLDEN" else mode
            item_adapt = adapt + [f"reference {ref_id}: " + (f"gt_answer={gt!r} (evidence-grounded reference, NOT verified ground truth)"
                                                             if gt else "no gt_answer; only faithfulness is defined")]
            status = ERROR if err else OK
            mv = []
            for m in METRICS:
                if mode == "NO_REFERENCE" and m != "faithfulness":
                    mv.append(MetricValue(self.name, mode, m, None, HIGHER_BETTER.get(m, True), "[0,1]", NOT_APPLICABLE,
                                          ["requires gt_answer"], REFERENCE_DEPENDENT[m]))
                    continue
                v = r.metrics.get(m) if not err else None
                mv.append(MetricValue(self.name, ref_id, m, None if v is None else float(v),
                                      HIGHER_BETTER.get(m, True), "[0,1] per item (RAGChecker reports x100 aggregated)", status,
                                      reference_dependent=REFERENCE_DEPENDENT[m]))
            raw = {"response_claims": r.response_claims, "gt_answer_claims": r.gt_answer_claims,
                   "retrieved2response": r.retrieved2response, "answer2response": r.answer2response,
                   "response2answer": r.response2answer, "retrieved2answer": r.retrieved2answer,
                   "gt_answer": gt, "reference_id": ref_id}
            out.append(EvaluationResult(self.name, ref_id, self.impl, item_adapt, self.judge.describe(), status, mv, raw,
                                        TEXT_ONLY_METADATA, rt, led.summary(), list(warns), err,
                                        config_hash([mode, self.impl, self.judge.describe(), self.joint_check])))
        return out
