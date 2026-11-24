"""ConflictScore adapter: the paper's response metric only.

published_cs : the PAPER's response metric (Section 3): Table-10 claim decomposition,
               Table-11 evidence labels, CS-C and CS-R (both lower-is-better).
               Prompts, parsing and aggregation live in conflictscore_published.py
               because the official repository (microsoft/ConflictScore) contains no
               CS-C/CS-R code; only its label parser is ported (see that module).

ADAPTATIONS (always reported): judge substituted (paper main results: Gemini-3.5-Flash;
repository default: gpt-4.1); temperature=0 not enforceable through the judge.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from ..judges.base import JudgeError
from . import conflictscore_published as pub
from .base import ERROR, OK, TEXT_ONLY_METADATA, EvaluationResult, EvaluatorAdapter, MetricValue, config_hash, git_sha


class ConflictScoreAdapter(EvaluatorAdapter):
    name = "ConflictScore"
    modes = ["published_cs"]

    def __init__(self, judge, max_workers: int = 4):
        super().__init__(judge)
        self.max_workers = max_workers
        self.sha = git_sha("ConflictScore")  # provenance of the ported label parser

    def _adaptations(self, mode="published_cs"):
        return [f"judge substituted: {self.judge.describe()} (paper main results: Gemini-3.5-Flash; "
                "repository default: gpt-4.1)",
                "temperature=0 requested by official code but not enforceable via judge backend",
                "prompts transcribed from paper Tables 10/11 (quotes/hyphenation normalised); "
                "no official code exists for CS-C/CS-R",
                "claim lines parsed with prefix 'Claims:' or 'Claim:'; fallback to whole answer",
                "CS-R excludes claims with no SUPPORTS/CONTRADICTS document (undefined in paper)",
                "judge errors reported as ERROR, not counted as IRRELEVANT"]

    def evaluate(self, query, answer, contexts, scenario_metadata, reference=None, mode="published_cs"):
        return self.evaluate_many([dict(item_id="single", query=query, answer=answer, contexts=contexts,
                                        scenario_metadata=scenario_metadata)], mode=mode)[0]

    def evaluate_many(self, items: List[Dict], mode: str = "published_cs") -> List[EvaluationResult]:
        if mode != "published_cs":
            raise ValueError(f"unsupported ConflictScore mode {mode!r}; only 'published_cs' (paper Sec. 3)")
        return self._published(items)

    # --------------------------------------------------- published CS-C/CS-R
    def _published(self, items):
        """Paper Section 3 (CS-C/CS-R) with the paper's Table 10/11 prompts. See conflictscore_published."""
        mode = "published_cs"

        def run(it):
            t0 = time.time()
            led = self.judge.open_ledger()
            errors, warns = [], []
            claims, claim_labels, responses = [], [], []
            try:
                try:
                    dec = self.judge.complete(pub.decomposition_prompt(it["answer"]), tag="conflictscore-published").text
                except JudgeError as e:
                    raise RuntimeError(f"decomposition failed: {e}")
                claims = pub.parse_claims(dec)
                if not claims:
                    claims = [it["answer"]]
                    warns.append("no 'Claims:'/'Claim:' line parsed; whole answer used as one claim")
                for c in claims:
                    labs, resps = [], []
                    for d in it["contexts"]:
                        try:
                            r = self.judge.complete(pub.evidence_prompt(c, d), tag="conflictscore-published").text
                        except JudgeError as e:
                            raise RuntimeError(f"evidence evaluation failed: {e}")
                        lab = pub.parse_label(r)
                        if lab not in pub.LABELS:
                            warns.append(f"unrecognised label {lab!r} counted as IRRELEVANT (official behaviour)")
                        labs.append(lab)
                        resps.append(r)
                    claim_labels.append(labs)
                    responses.append(resps)
            except RuntimeError as e:
                errors.append(str(e))
            finally:
                self.judge.close_ledger(led)
            if led.n_errors and not errors:
                errors.append(f"{led.n_errors} judge errors")
            status = ERROR if errors else OK
            cs = pub.cs_metrics(claim_labels) if status == OK else {}
            if status == OK and cs["cs_r"] is None:
                warns.append("CS-R undefined: no claim has a SUPPORTS or CONTRADICTS document")
            g = lambda k: cs.get(k) if status == OK else None
            mv = [MetricValue(self.name, mode, "cs_c", g("cs_c"), False,
                              "[0,1]; fraction of claims with >=1 SUPPORTS and >=1 CONTRADICTS doc (paper CS-C)", status),
                  MetricValue(self.name, mode, "cs_r", g("cs_r"), False,
                              "[0,1]; mean |D-|/(|D+|+|D-|) over claims with relevant docs (paper CS-R)", status),
                  MetricValue(self.name, mode, "n_claims", g("n_claims"), None, "count", status),
                  MetricValue(self.name, mode, "n_conflicted_claims", g("n_conflicted_claims"), None, "count", status),
                  MetricValue(self.name, mode, "n_claims_without_relevant_evidence",
                              g("n_claims_without_relevant_evidence"), None, "count", status)]
            raw = {"claims": claims, "claim_doc_labels": claim_labels, "label_responses": responses, "cs": cs}
            return EvaluationResult(self.name, mode, f"paper:2026.emnlp-main.2589 (Tables 10-11, Sec.3); "
                                    f"label parser from microsoft/ConflictScore@{self.sha}",
                                    self._adaptations(mode), self.judge.describe(), status, mv, raw,
                                    TEXT_ONLY_METADATA, time.time() - t0, led.summary(), warns,
                                    "; ".join(errors) or None, config_hash([mode, self.sha, self.judge.describe()]))

        with ThreadPoolExecutor(self.max_workers) as ex:
            return list(ex.map(run, items))
