"""Methodology tests: published formulas and definitions, checked against the
papers' own worked examples and against independent re-implementations of the published
formulas. No network, no real judge."""

import itertools
import json
import random
import re

import numpy as np
import pytest

from benchmark import data_factory as df
from benchmark.evaluators import conflictscore_published as pub
from benchmark.judges.base import Judge, JudgeError, JudgeResponse

S, C, I = "SUPPORTS", "CONTRADICTS", "IRRELEVANT"


# ================================================================ ConflictScore
def test_cs_paper_figure2_example():
    """Paper Fig. 2: six claims over four docs -> CS-C 0.67, CS-R 0.38 (printed, rounded)."""
    claims = [[S, S, S, S], [S, S, S, C], [S, S, C, C], [S, C, C, C], [S, C, C, C], [S, I, I, I]]
    m = pub.cs_metrics(claims)
    assert m["cs_c"] == pytest.approx(4 / 6) and round(m["cs_c"], 2) == 0.67
    assert m["cs_r"] == pytest.approx(0.375) and round(m["cs_r"] + 1e-9, 2) == 0.38
    assert m["n_conflicted_claims"] == 4 and m["n_claims_without_relevant_evidence"] == 0


def test_cs_paper_figure5_example():
    """Paper Fig. 5: one claim, D+ = 1, D- = 5 -> CS-C 1.00, CS-R 5/6 = 0.83."""
    m = pub.cs_metrics([[S, C, C, C, C, C]])
    assert m["cs_c"] == 1.0 and round(m["cs_r"], 2) == 0.83


def test_cs_conflict_requires_both_support_and_contradiction():
    """Paper Sec. 3 / 4.1: conflicted iff D+ and D- both non-empty (S=0, C=2 is NOT a conflict)."""
    assert not pub.is_conflicted([C, C])
    assert not pub.is_conflicted([S, S])
    assert pub.is_conflicted([S, C])
    assert pub.contradiction_ratio([C, C]) == 1.0   # maximal contradiction, but not "conflict"


def test_cs_r_undefined_claims_excluded_but_counted_in_cs_c():
    m = pub.cs_metrics([[S, C], [I, I]])
    assert m["cs_c"] == 0.5                 # denominator = all claims (as in Fig. 2)
    assert m["cs_r"] == 0.5                 # [I, I] excluded from CS-R mean
    assert m["n_claims_without_relevant_evidence"] == 1
    assert pub.cs_metrics([[I, I]])["cs_r"] is None


def test_cs_direction_is_lower_is_better():
    from benchmark.metric_defs import DEFS
    assert DEFS[("ConflictScore", "cs_c")][1] is False and DEFS[("ConflictScore", "cs_r")][1] is False


def test_published_prompts_match_paper_not_repository():
    p = pub.evidence_prompt("CLAIM", "DOC")
    # paper Table 11: no question field; repository prompt has '# Question'
    assert "# Question" not in p and "{question}" not in pub.EVIDENCE_PROMPT
    # repository-only rule absent; paper-specific wording present
    assert "If both supporting and contradicting statements appear" not in p
    assert "directly supports any part of the claim" in p
    assert "Look for statements in the document that address the same attributes asserted" in p
    assert "# Claim to evaluate\nCLAIM" in p and "# Document to evaluate against\nDOC" in p
    d = pub.decomposition_prompt("REPORT")
    assert d.startswith("Instruction: Break down the following report into individual claims.")
    assert 'starting with "Claims: "' in d and "Question" not in d


def test_published_prompt_differs_from_repo_prompt():
    """Guard against silently reusing the repository's calibration prompt."""
    import runpy, contextlib, io
    from benchmark.evaluators.base import EXTERNAL
    REPO = EXTERNAL / "ConflictScore"
    with contextlib.redirect_stdout(io.StringIO()):
        mod = runpy.run_path(str(REPO / "src/eval_with_conflict_score/eval_all_calibration.py"), run_name="x")
    seen = {}

    class Grab:
        class chat:
            class completions:
                @staticmethod
                def create(model=None, messages=None, **k):
                    seen["p"] = messages[0]["content"]
                    raise RuntimeError("stop")
    mod["evaluate_document_support"](Grab, "DOC", "CLAIM", "QUESTION", "m")
    assert "# Question\nQUESTION" in seen["p"] and seen["p"] != pub.evidence_prompt("CLAIM", "DOC")


@pytest.mark.parametrize("text,label", [
    ('{"claim":"x","document_snippet":"y","reasoning":"z","answer":"supports"}', S),
    ('```json\n{"answer": "CONTRADICTS"}\n```', C),
    ('{"answer": "CONTRADICTS", "reasoning": "broken" ', C),        # malformed -> no JSON match -> tag/none
    ('blah <answer>CONTRADICTS</answer>', C),
    ('no label at all', I),
])
def test_label_parser_port(text, label):
    got = pub.parse_label(text)
    if text.startswith('{"answer": "CONTRADICTS", "reasoning": "broken"'):
        # official behaviour: regex finds no complete object and no <answer> tag -> IRRELEVANT
        assert got == I
    else:
        assert got == label


def test_claims_parser():
    t = "Claims: A gives 3 March.\nClaims: B gives 17 March.\n\nnoise\nClaim: single\n1. Claims: numbered\n- Claims: bullet"
    assert pub.parse_claims(t) == ["A gives 3 March.", "B gives 17 March.", "single", "numbered", "bullet"]
    assert pub.parse_claims("nothing here") == []


class PaperScripted(Judge):
    """Rule-based stand-in for plumbing tests only."""
    backend, model = "scripted-test", "scripted"

    def __init__(self, fail_on=None):
        super().__init__()
        self.fail_on = fail_on
        self.prompts = []

    def _complete(self, prompt, system=None, tag="", output_schema=None):
        self.prompts.append(prompt)
        if self.fail_on and self.fail_on in prompt:
            raise JudgeError("scripted failure")
        if prompt.startswith("Instruction: Break down the following report"):
            rep = re.search(r"Report: (.*)", prompt).group(1)
            text = "\n".join(f"Claims: {s.strip()}." for s in rep.rstrip(".").split(", while "))
        else:
            claim = prompt.split("# Claim to evaluate\n")[1].split("\n\n# Document")[0]
            doc = prompt.split("# Document to evaluate against\n")[1].split("\n\n# Output format")[0]
            dc, dd = re.findall(r"(\d+) March", claim), re.findall(r"(\d+) March", doc)
            lab = I if not dc else (S if dc == dd else C)
            text = json.dumps({"claim": claim, "document_snippet": doc, "reasoning": "r", "answer": lab})
        return JudgeResponse(text, self.model, self.backend, False, 0.0, {}, "t")


def _c1_items(ids=("R1", "R5")):
    sc = {s["scenario_id"]: s for s in df.build_scenarios()}["C1"]
    ctx = [df.render_doc(d) for d in sc["documents"]]
    return [dict(item_id=r["response_id"], query=sc["question"], answer=r["text"], contexts=ctx, scenario_metadata={})
            for r in df.RESPONSES if r["response_id"] in ids]


def _vals(res):
    return {m.native_metric_name: m.native_metric_value for m in res.metrics}


def test_published_mode_end_to_end():
    from benchmark.evaluators.conflictscore import ConflictScoreAdapter
    j = PaperScripted()
    r1, r5 = ConflictScoreAdapter(j).evaluate_many(_c1_items(), mode="published_cs")
    assert r1.status == "OK" and r1.raw["claim_doc_labels"] == [[S, C]]
    assert _vals(r1)["cs_c"] == 1.0 and _vals(r1)["cs_r"] == 0.5
    # R5 split into 'Archive A gives 3 March 1990' / 'Registry B gives 17 March 1990. Both agree ...'
    assert _vals(r5)["n_claims"] == 2
    assert all("Question" not in p for p in j.prompts)       # paper prompts never see the question
    assert r1.mode == "published_cs" and any("Tables 10/11" in a for a in r1.adaptation)


def test_published_mode_judge_error_is_error_not_irrelevant():
    from benchmark.evaluators.conflictscore import ConflictScoreAdapter
    res = ConflictScoreAdapter(PaperScripted(fail_on="Registry B")).evaluate_many(_c1_items(("R1",)), mode="published_cs")
    assert res[0].status == "ERROR" and _vals(res[0])["cs_c"] is None


def test_only_published_mode_is_supported():
    from benchmark.evaluators.conflictscore import ConflictScoreAdapter
    assert ConflictScoreAdapter.modes == ["published_cs"]
    with pytest.raises(ValueError):
        ConflictScoreAdapter(PaperScripted()).evaluate_many(_c1_items(("R1",)), mode="conflictbench_protocol")


def test_adaptation_text_names_correct_paper_judge():
    from benchmark.evaluators.conflictscore import ConflictScoreAdapter
    a = ConflictScoreAdapter(PaperScripted())._adaptations("published_cs")[0]
    assert "Gemini-3.5-Flash" in a and "repository default: gpt-4.1" in a


# ================================================================ RAGChecker
def _paper_formulas(A, B, E, G):
    """Independent implementation of RAGChecker paper Appendix B (claim-level indicators)."""
    p, q, K = len(A), len(B), E.shape[1]
    rel = G.max(axis=0).astype(bool)                      # chunk relevant iff it entails >=1 gt claim
    faithful = E.max(axis=1).astype(bool)
    in_rel = (E[:, rel].max(axis=1) if rel.any() else np.zeros(p)).astype(bool)
    in_irr = (E[:, ~rel].max(axis=1) if (~rel).any() else np.zeros(p)).astype(bool) & ~in_rel
    correct = A.astype(bool)
    recalled = G.max(axis=1).astype(bool)
    P, R = A.mean(), B.mean()
    return {
        "precision": P, "recall": R, "f1": (2 * P * R / (P + R)) if P > 0 and R > 0 else 0.0,
        "claim_recall": recalled.mean(), "context_precision": rel.mean(),
        "context_utilization": (recalled & B.astype(bool)).sum() / recalled.sum() if recalled.sum() else 0.0,
        "noise_sensitivity_in_relevant": (~correct & in_rel).mean(),
        "noise_sensitivity_in_irrelevant": (~correct & in_irr).mean(),
        "hallucination": (~correct & ~faithful).mean(), "self_knowledge": (correct & ~faithful).mean(),
        "faithfulness": faithful.mean(),
    }


def _official(A, B, E, G, labels=("Neutral", "Entailment")):
    from ragchecker.container import RAGResult
    from ragchecker.computation import METRIC_FUNC_MAP
    lab = lambda x: "Entailment" if x else "Neutral"
    r = RAGResult(query_id="q", query="q", gt_answer="g", response="r", retrieved_context=[])
    r.answer2response = [lab(x) for x in A]
    r.response2answer = [lab(x) for x in B]
    r.retrieved2response = [[lab(x) for x in row] for row in E]
    r.retrieved2answer = [[lab(x) for x in row] for row in G]
    for f in METRIC_FUNC_MAP.values():
        f(r)
    return r.metrics


def test_ragchecker_package_matches_paper_formulas_randomised():
    rng = random.Random(0)
    for _ in range(300):
        p, q, K = rng.randint(1, 5), rng.randint(1, 5), rng.randint(1, 4)
        A = np.array([rng.random() < .5 for _ in range(p)], int)
        B = np.array([rng.random() < .5 for _ in range(q)], int)
        E = np.array([[rng.random() < .4 for _ in range(K)] for _ in range(p)], int)
        G = np.array([[rng.random() < .4 for _ in range(K)] for _ in range(q)], int)
        off, ref = _official(A, B, E, G), _paper_formulas(A, B, E, G)
        for k, v in ref.items():
            assert off[k] == pytest.approx(v), k


def test_ragchecker_c1_r5_worked_example():
    """C1|Q_DATE|R5 judgments (codex-luna) -> faithfulness 0.75, self_knowledge 0.25."""
    from ragchecker.container import RAGResult
    from ragchecker.computation import METRIC_FUNC_MAP
    r = RAGResult(query_id="q", query="q", gt_answer="g", response="r", retrieved_context=[])
    r.retrieved2response = [["Entailment", "Neutral"], ["Neutral", "Entailment"], ["Entailment", "Neutral"], ["Neutral", "Neutral"]]
    r.answer2response = ["Entailment"] * 4
    r.response2answer = ["Entailment"] * 4
    r.retrieved2answer = [["Entailment", "Neutral"], ["Neutral", "Entailment"], ["Neutral", "Neutral"], ["Neutral", "Contradiction"]]
    for f in METRIC_FUNC_MAP.values():
        f(r)
    assert r.metrics["faithfulness"] == 0.75 and r.metrics["self_knowledge"] == 0.25
    assert r.metrics["claim_recall"] == 0.5 and r.metrics["context_utilization"] == 1.0


def test_ragchecker_contradiction_counts_as_not_entailed():
    """R1 (Q_DATE): [[Entailment, Contradiction]] -> faithfulness 1.0 (max over chunks)."""
    from ragchecker.computation import to_bool
    assert list(to_bool(["Entailment", "Contradiction", "Neutral"])) == [True, False, False]


def test_ragchecker_reference_dependence_derived_from_package():
    """A metric is reference-dependent iff the package requires a check involving the gt answer."""
    from ragchecker.metrics import METRIC_REQUIREMENTS
    from benchmark.evaluators.ragchecker import REFERENCE_DEPENDENT
    gt_checks = {"answer2response", "response2answer", "retrieved2answer"}
    for m, reqs in METRIC_REQUIREMENTS.items():
        assert REFERENCE_DEPENDENT[m] == bool(gt_checks & set(reqs)), m
    assert REFERENCE_DEPENDENT["faithfulness"] is False


# ================================================================ Q-CARE
def test_qcare_c1_r5_worked_example():
    import sys
    from benchmark.evaluators.qcare import REPO
    sys.path.insert(0, str(REPO))
    from qcare.metrics import compute_metrics
    rc = {"query_fact_relevance": {"Core subquery1": {"selected_facts": ["Atomic fact1", "Atomic fact2"]}},
          "chunk_fact_relevance": {"Chunk 1": {"selected_facts": ["Atomic fact1"]}, "Chunk 2": {"selected_facts": ["Atomic fact2"]}},
          "query_chunk_coverage": {"Chunk 1": ["Core subquery1"], "Chunk 2": ["Core subquery1"]},
          "query_fact_coverage": {"Core subquery1": "Covered"}}
    m = compute_metrics({"Core subquery1": "q"}, {f"Atomic fact{i}": "f" for i in (1, 2, 3)}, rc)
    assert m["completeness"] == 1.0 and m["conciseness"] == pytest.approx(2 / 3)
    assert m["verifiableness"] == pytest.approx(2 / 3)
    assert m["precision_at_10"] == pytest.approx(0.2) and m["ndcg_at_10"] == pytest.approx(1.0)


def test_qcare_backbone_substitution_passes_messages_unchanged():
    from benchmark.evaluators.qcare import JudgeBackbone

    class Rec(PaperScripted):
        def chat(self, messages, tag=""):
            self.got = messages
            return JudgeResponse("Yes", "s", "s", False, 0, {}, "t")
    j = Rec()
    msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}]
    assert JudgeBackbone(j).generate(msgs, max_new_tokens=5, temperature=0.0) == "Yes"
    assert j.got == msgs


# ================================================================ judges
def test_codex_model_identity_labelled_unverified(tmp_path):
    from benchmark.judges.codex import CodexJudge
    j = CodexJudge(codex_bin="/nonexistent", cache_dir=str(tmp_path))
    j._version = "x"
    assert "UNVERIFIED" in j.describe()["model_identity"]


def test_parse_claims_accepts_header_then_list():
    """Table 10 asks to list each claim 'starting with "Claims: "'. Some judges write one header line and
    then one claim per line (observed with claude-opus-5-5); those claims must not be discarded."""
    from benchmark.evaluators.conflictscore_published import parse_claims
    opus = ("Claims:\nArchive A gives the date as 3 March 1990.\nRegistry B gives the date as 17 March 1990.\n"
            "Archive A and Registry B both agree that the date is in March 1990.\nThe precise day of the date is disputed.")
    assert parse_claims(opus) == ["Archive A gives the date as 3 March 1990.",
                                  "Registry B gives the date as 17 March 1990.",
                                  "Archive A and Registry B both agree that the date is in March 1990.",
                                  "The precise day of the date is disputed."]
    assert parse_claims("Claims:\n- first claim\n2. second claim\n\n") == ["first claim", "second claim"]
    # the per-line form keeps its exact behaviour
    assert parse_claims("Claims: A.\nClaims: B.") == ["A.", "B."]
    assert parse_claims("Here are the claims\nClaim: A.") == ["A."]
    assert parse_claims("no claims here") == []
