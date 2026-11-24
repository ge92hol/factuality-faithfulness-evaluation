"""Q-CARE adapter (DISL-Lab/Q-CaRE-COLM-26, MIT).

Uses the official `QCAREPipeline` (decomposition, four alignment checks, metrics,
`parsing="paper"`). Only the `Backbone` object is substituted: the official class
loads a local HF model (default Qwen3-30B-A3B, ~70 GB bf16 — not feasible on the
6 GB GPU here); ours forwards the same chat messages (system + user, from
configs/prompts.yaml) to the judge, with the system message as the judge's
system instructions.

Native metrics: completeness, conciseness, verifiableness (generator);
precision_at_10 (C-Prec@10) and ndcg_at_10 (C-nDCG@10) (retriever). Note C-Prec@10
divides by k=10, so with 2 oracle documents its maximum is 0.2 by definition.

ADAPTATIONS: backbone substituted (API/CLI model; the README says the backbone must
be an open-weight HF model loaded with transformers — this is therefore NOT a
supported configuration of the original code); greedy decoding not enforceable;
transformers 4.57 instead of pinned 5.7.0 (only affects the unused local loader).
Replacing `Backbone` also replaces (i) its 3-attempt retry loop -> retries now come
from the judge backend (CodexJudge max_retries=2, i.e. 3 attempts; quota errors are
not retried), and (ii) its max_new_tokens=2000 cap -> not forwarded. Audit 2026-11-16:
the largest visible Q-CARE judge output in the cache is 94 tokens (codex) / 224
(claude, incl. hidden reasoning), so the cap was never binding.
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from .base import (ERROR, EXTERNAL, OK, TEXT_ONLY_METADATA, EvaluationResult, EvaluatorAdapter,
                   MetricValue, config_hash, git_sha)

REPO = EXTERNAL / "Q-CaRE-COLM-26"
NATIVE = [("completeness", True), ("conciseness", True), ("verifiableness", True),
          ("precision_at_10", True), ("ndcg_at_10", True)]


class JudgeBackbone:
    """Duck-types qcare.backbone.Backbone.generate(messages, max_new_tokens, temperature, max_retries)."""

    def __init__(self, judge):
        self.judge = judge
        self.model_name = f"{judge.backend}:{getattr(judge, 'model', '?')}"

    def load(self):
        return self

    def generate(self, messages, max_new_tokens=2000, temperature=0.0, max_retries=3):
        return self.judge.chat(messages, tag="qcare").text


class QCAREAdapter(EvaluatorAdapter):
    name = "Q-CARE"
    modes = ["paper"]

    def __init__(self, judge, max_workers: int = 4, parsing: str = "paper"):
        super().__init__(judge)
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        from qcare import QCAREPipeline, PromptLibrary
        self.pipeline = QCAREPipeline(JudgeBackbone(judge), PromptLibrary(str(REPO / "configs/prompts.yaml")),
                                      parsing=parsing)
        self.parsing = parsing
        self.max_workers = max_workers
        self.sha = git_sha("Q-CaRE-COLM-26")

    def evaluate(self, query, answer, contexts, scenario_metadata, reference=None, mode="paper"):
        return self.evaluate_many([dict(item_id="single", query=query, answer=answer, contexts=contexts,
                                        scenario_metadata=scenario_metadata)])[0]

    def evaluate_many(self, items: List[Dict], mode: str = "paper") -> List[EvaluationResult]:
        adapt = [f"backbone substituted: {self.judge.describe()} (repository default: Qwen/Qwen3-30B-A3B-Instruct-2507, "
                 "local HF; paper also evaluates Qwen3-80B, Llama3.1-8B, Llama3.3-70B per README)",
                 "official Backbone.generate retry loop and max_new_tokens=2000 replaced by judge backend "
                 "(retries: judge max_retries; no output cap forwarded)",
                 "README states backbone must be an open-weight HF chat model; CLI judge is outside supported config",
                 "greedy decoding (temperature=0) not enforceable via judge backend",
                 "transformers 4.57.6 instead of pinned 5.7.0 (local loader unused)"]

        def run(it):
            t0 = time.time()
            led = self.judge.open_ledger()
            err, res = None, None
            try:
                res = self.pipeline.evaluate(it["item_id"], it["query"], it["answer"], list(it["contexts"]))
            except Exception as e:  # noqa: BLE001  (JudgeError after the JUDGE's retries; the official
                #                     Backbone retry loop is bypassed by JudgeBackbone)
                err = f"{type(e).__name__}: {e}"
            finally:
                self.judge.close_ledger(led)
            status = ERROR if err else OK
            m = (res or {}).get("metrics", {})
            mv = [MetricValue(self.name, f"parsing={self.parsing}", k, None if err else m.get(k), hib,
                              "[0,1]" + ("; max=n_chunks/10" if k == "precision_at_10" else ""), status)
                  for k, hib in NATIVE]
            mv.append(MetricValue(self.name, f"parsing={self.parsing}", "num_atomic_facts",
                                  None if err else len(res["atomic_facts"]), None, "count", status))
            raw = {k: (res or {}).get(k) for k in ("decomposed_query", "atomic_facts", "relevance_check", "metrics")}
            return EvaluationResult(self.name, f"parsing={self.parsing}", f"DISL-Lab/Q-CaRE-COLM-26@{self.sha}", adapt,
                                    self.judge.describe(), status, mv, raw, TEXT_ONLY_METADATA, time.time() - t0,
                                    led.summary(), [], err, config_hash(["qcare", self.sha, self.parsing, self.judge.describe()]))

        with ThreadPoolExecutor(self.max_workers) as ex:
            return list(ex.map(run, items))
