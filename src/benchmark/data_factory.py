"""Controlled birth-date dataset (spec §4, §15).

Everything here is deterministic and version-controlled. Reliability values are
ASSIGNED TEST VALUES (high=0.9, low=0.3) for controlled experiments only; they
are not empirical reliability estimates. There is no independent gold birthday
for the fictional entity: the field reference records what each source says.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Dict, List

ENTITY = "Alex Meron"
RELIABILITY = {"high": 0.9, "low": 0.3}  # ASSIGNED TEST VALUES

QUESTIONS = {
    "Q_CANON": "When was the fictional person Alex Meron born?",
    "Q_YEAR": "Which year was Alex Meron born?",
    "Q_DATE": "What exact date was Alex Meron born?",
}

# --- fixed candidate responses (spec §4: preserve exactly) ---------------
RESPONSES = [
    {"response_id": "R1", "label": "EXACT-A", "main": True,
     "text": "Alex Meron was born on 3 March 1990."},
    {"response_id": "R2", "label": "MONTH", "main": True,
     "text": "Alex Meron was born in March 1990."},
    {"response_id": "R3", "label": "YEAR", "main": True,
     "text": "Alex Meron was born in 1990."},
    {"response_id": "R4", "label": "VAGUE", "main": True,
     "text": "The exact date of birth is uncertain."},
    {"response_id": "R5", "label": "ATTRIBUTED-CONFLICT", "main": True,
     "text": "Archive A gives 3 March 1990, while Registry B gives 17 March 1990. "
             "Both agree on March 1990, but the precise day is disputed."},
    # additional variants (spec §4)
    {"response_id": "V1", "label": "HEDGED-EXACT-A", "main": False,
     "text": "Alex Meron was probably born on 3 March 1990."},
    {"response_id": "V2", "label": "EXACT-B", "main": False,
     "text": "Alex Meron was born on 17 March 1990."},
    {"response_id": "V3", "label": "UNSUPPORTED", "main": False,
     "text": "Alex Meron was born on 9 April 1992."},
    {"response_id": "V4", "label": "FALSE-CONSENSUS", "main": False,
     "text": "Both sources agree on 3 March 1990."},
]

FIELD_REFERENCE = {
    "year": {"value": 1990, "status": "shared_by_A_and_B"},
    "month": {"value": "March", "status": "shared_by_A_and_B"},
    "day": {"A": 3, "B": 17, "status": "contested"},
    "independent_gold": None,  # NEVER inferred from reliability or majority
    "flags": ["contested_day", "shared_month_year"],
}

# ---------------------------------------------------------------------------
# EVIDENCE-GROUNDED REFERENCE ANSWERS (spec: "Golden Reference Answers for RAGChecker")
# Fixed per (scenario, question) BEFORE any candidate is scored; identical for R1-R5.
# They describe what an ideal answer can claim FROM THE RETRIEVED EVIDENCE. They are
# NOT independently verified factual ground truth (none exists for this fictional
# entity; FIELD_REFERENCE['independent_gold'] is None).
#
# Construction principles (written before scoring; not tuned to any candidate):
#  P1 Answer at the granularity the question requests, using only evidence content.
#  P2 State fields on which all retrieved sources agree plainly.
#  P3 State disagreement at the requested granularity with source attribution and
#     without choosing a side when sources have equal assigned reliability.
#  P4 Unequal ASSIGNED reliability: name the source the assignment favours, still
#     report the dissenting source, and do not declare the conflict resolved.
#  P5 A source hedge ("probably") is reported as that source's tentative statement; under
#     equal assigned reliability the definite report is identified as better supported,
#     while the conflicting tentative report is still stated (C4/C5; revised 2026-11-23).
#  P6 Disagreement below the requested granularity is immaterial (year question:
#     both sources say 1990, so the day dispute is not part of the reference).
#  P7 Order reversal / duplication do not change evidence content -> same reference as C1.
GOLDEN_C1_DATE = ("The retrieved records disagree about Alex Meron's exact birth date. "
                  "Archive A reports 3 March 1990, whereas Registry B reports 17 March 1990. "
                  "Both agree on March 1990. The available evidence does not establish which day is correct.")
_YEAR_REF = "Alex Meron was born in 1990."

GOLDEN_REFERENCES = {
    # (scenario, question_id): (reference_text, construction note)
    ("C1", "Q_DATE"): (GOLDEN_C1_DATE, "verbatim from spec golden section (user-authored)"),
    ("C1", "Q_YEAR"): (_YEAR_REF, "P1,P2,P6; wording suggested by user"),
    ("C2", "Q_DATE"): ("The retrieved records disagree about Alex Meron's exact birth date. Archive A, the source "
                       "assigned higher reliability, reports 3 March 1990, whereas Registry B, assigned lower "
                       "reliability, reports 17 March 1990. Both agree on March 1990. The evidence favours "
                       "3 March 1990 but does not conclusively establish which day is correct.", "P2,P3,P4"),
    ("C3", "Q_DATE"): ("The retrieved records disagree about Alex Meron's exact birth date. Registry B, the source "
                       "assigned higher reliability, reports 17 March 1990, whereas Archive A, assigned lower "
                       "reliability, reports 3 March 1990. Both agree on March 1990. The evidence favours "
                       "17 March 1990 but does not conclusively establish which day is correct.", "P2,P3,P4 (mirror of C2)"),
    ("C4", "Q_DATE"): ("Archive A reports that Alex Meron was born on 3 March 1990, whereas Registry B states that "
                       "he was probably born on 17 March 1990. Both agree on March 1990. Given their equal assigned "
                       "reliability, 3 March 1990 is better supported by the supplied evidence because Archive A "
                       "reports it without qualification, although a conflicting tentative report remains.", "P2,P5"),
    ("C5", "Q_DATE"): ("Registry B reports that Alex Meron was born on 17 March 1990, whereas Archive A states that "
                       "he was probably born on 3 March 1990. Both agree on March 1990. Given their equal assigned "
                       "reliability, 17 March 1990 is better supported by the supplied evidence because Registry B "
                       "reports it without qualification, although a conflicting tentative report remains.", "P2,P5 (mirror of C4)"),
    ("NC", "Q_DATE"): ("Alex Meron was born on 3 March 1990.", "P1,P2: both sources agree"),
    ("ORD_BA", "Q_DATE"): (GOLDEN_C1_DATE, "P7"),
    ("DUP_A3", "Q_DATE"): (GOLDEN_C1_DATE, "P7 (duplicates are not independent evidence)"),
}
for _s in ("C2", "C3", "C4", "C5", "NC", "ORD_BA", "DUP_A3"):
    GOLDEN_REFERENCES[(_s, "Q_YEAR")] = (_YEAR_REF, "P1,P2,P6")

# Alternative references for REFERENCE-SENSITIVITY probes only (C1, Q_DATE). Labelled
# as probes; not evidence-grounded ideals under equal reliability (REF_A/REF_B).
ALT_REFERENCES = {
    "ALT_MONTH": ("Alex Meron was born in March 1990; the exact day is not established by the evidence.",
                  "defensible: shared information only"),
    "ALT_REF_A": ("Alex Meron was born on 3 March 1990.", "probe: single-source reference (Archive A)"),
    "ALT_REF_B": ("Alex Meron was born on 17 March 1990.", "probe: single-source reference (Registry B)"),
}


def golden_reference(scenario_id: str, question_id: str) -> Dict:
    base = scenario_id.split("_Q_")[0] if "_Q_" in scenario_id else scenario_id
    text, note = GOLDEN_REFERENCES[(base, question_id)]
    return {"reference_id": f"GOLD:{base}:{question_id}", "text": text, "construction": note,
            "kind": "evidence-grounded ideal response (NOT verified ground truth)"}


# Legacy experimental reference modes (smoke test); kept for reproducibility.
REFERENCES = {
    "REF_A": "Alex Meron was born on 3 March 1990.",
    "REF_B": "Alex Meron was born on 17 March 1990.",
    "SHARED_MONTH": "Alex Meron was born in March 1990.",
    "CONFLICT_AWARE": "Archive A gives 3 March 1990 and Registry B gives 17 March 1990; "
                      "both agree on March 1990, and the exact day is disputed.",
}


def _doc(doc_id: str, source_name: str, day: int, rel: str, hedge: str | None) -> Dict:
    verb = "was probably born on" if hedge == "probably" else "was born on"
    return {
        "doc_id": doc_id,
        "source_id": f"src_{doc_id}",
        "source_name": source_name,
        "source_group": f"group_{doc_id}",
        "url": f"synthetic://controlled/{doc_id}",
        "provenance": "synthetic controlled fixture (spec §4)",
        "timestamp": None,
        "content": f"{ENTITY} {verb} {day} March 1990.",
        "stance": {"day": day, "month": "March", "year": 1990},
        "reliability_label": rel,
        "oracle_reliability": RELIABILITY[rel],
        "reliability_method": "oracle synthetic (assigned test value)",
        "estimated_reliability": None,
        "source_hedge": hedge,  # None = definite
    }


def A(rel="high", hedge=None, day=3):
    return _doc("A", "Archive A", day, rel, hedge)


def B(rel="high", hedge=None, day=17):
    return _doc("B", "Registry B", day, rel, hedge)


def build_scenarios() -> List[Dict]:
    """Each scenario changes exactly one variable relative to C1 (noted in `manipulated`)."""
    base = [
        ("C1", "symmetric high reliability, both definite", [A(), B()], "none (reference condition)"),
        ("C2", "asymmetric reliability: B low", [A(), B(rel="low")], "reliability(B)"),
        ("C3", "asymmetric reliability: A low", [A(rel="low"), B()], "reliability(A)"),
        ("C4", "source hedge in B ('probably')", [A(), B(hedge="probably")], "source_hedge(B)"),
        ("C5", "source hedge in A ('probably')", [A(hedge="probably"), B()], "source_hedge(A)"),
        ("NC", "no-conflict control: both sources say 3 March 1990", [A(), B(day=3)], "day(B)=3"),
        ("ORD_BA", "C1 with retrieval order reversed (B, A)", [B(), A()], "order"),
        ("DUP_A3", "C1 with Doc A repeated 3x under the same source id", [A(), A(), A(), B()], "duplication(A)x3"),
    ]
    out = []
    for sid, desc, docs, manip in base:
        docs = copy.deepcopy(docs)
        for i, d in enumerate(docs):
            d["position"] = i
            if sid == "DUP_A3" and d["doc_id"] == "A":
                d["chunk_id"] = f"A#copy{i}"  # same doc_id/source_id: correlated, not independent
            else:
                d["chunk_id"] = d["doc_id"]
        out.append({"scenario_id": sid, "description": desc, "manipulated": manip,
                    "question_id": "Q_CANON", "question": QUESTIONS["Q_CANON"], "documents": docs,
                    "field_reference": FIELD_REFERENCE})
    # question-intent variants on identical C1 evidence (spec §15 essential addition)
    c1 = out[0]
    for qid in ("Q_YEAR", "Q_DATE"):
        s = copy.deepcopy(c1)
        s.update(scenario_id=f"C1_{qid}", question_id=qid, question=QUESTIONS[qid],
                 description=f"C1 evidence, question wording {qid}", manipulated="question_wording")
        out.append(s)
    return out


def grid(scenario_ids, question_ids) -> List[Dict]:
    """Scenario instances with the question replaced; evidence held identical."""
    base = {s["scenario_id"]: s for s in build_scenarios()}
    out = []
    for sid in scenario_ids:
        for qid in question_ids:
            s = copy.deepcopy(base[sid])
            s.update(instance_id=f"{sid}|{qid}", question_id=qid, question=QUESTIONS[qid])
            s["reference"] = golden_reference(sid, qid) if qid in ("Q_YEAR", "Q_DATE") else None
            out.append(s)
    return out


# --- context rendering -----------------------------------------------------
# What text an evaluator sees for each document. Published evaluators accept
# only text; metadata reaches them only if it is rendered into that text.
RENDERINGS = {
    "attributed": "source name prefixed so attribution claims (R5) are checkable [DEFAULT]",
    "plain": "document content only",
    "attributed_reliability": "ADAPTED: assigned reliability label rendered into text (T2-text variant)",
}


def render_doc(d: Dict, rendering: str = "attributed") -> str:
    if rendering == "plain":
        return d["content"]
    if rendering == "attributed":
        return f"[Source: {d['source_name']}] {d['content']}"
    if rendering == "attributed_reliability":
        return f"[Source: {d['source_name']} | assigned reliability: {d['reliability_label']}] {d['content']}"
    raise ValueError(rendering)


def corpus_documents() -> List[Dict]:
    """Distinct-document corpus for retrieval tests: all scenario docs + distractors."""
    distractors = [
        ("D1", "Gazette C", "Alex Meron graduated from the Lindqvist Institute in 2012."),
        ("D2", "Archive A", "Archive A is a digitised collection of municipal records."),
        ("D3", "Registry B", "Registry B lists civil registrations for the northern district."),
        ("D4", "Gazette C", "Alexandra Meroni was born on 5 May 1987 in Turin."),
        ("D5", "Blog D", "Meron is a surname that appears in several regional archives."),
        ("D6", "Gazette C", "In March 1990 the northern district opened a new maternity ward."),
    ]
    docs = []
    for did, src, text in distractors:
        docs.append({"doc_id": did, "source_id": f"src_{did}", "source_name": src, "source_group": f"group_{did}",
                     "url": f"synthetic://distractor/{did}", "provenance": "synthetic distractor",
                     "timestamp": None, "content": text, "stance": None, "reliability_label": "high",
                     "oracle_reliability": RELIABILITY["high"], "reliability_method": "oracle synthetic (assigned test value)",
                     "estimated_reliability": None, "source_hedge": None})
    return docs


def write_controlled(out_dir: str) -> Dict[str, str]:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    files = {
        "responses.jsonl": RESPONSES,
        "scenarios.jsonl": build_scenarios(),
        "distractors.jsonl": corpus_documents(),
        "references.jsonl": [{"reference_id": f"GOLD:{k[0]}:{k[1]}", "scenario": k[0], "question_id": k[1],
                              "question": QUESTIONS[k[1]], "text": v[0], "construction": v[1],
                              "kind": "evidence-grounded ideal (NOT verified ground truth)"}
                             for k, v in sorted(GOLDEN_REFERENCES.items())]
                            + [{"reference_id": k, "scenario": "C1", "question_id": "Q_DATE", "text": v[0],
                                "construction": v[1], "kind": "reference-sensitivity probe"} for k, v in ALT_REFERENCES.items()]
                            + [{"reference_id": f"LEGACY:{k}", "text": v, "kind": "smoke-test mode"} for k, v in REFERENCES.items()],
    }
    hashes = {}
    for name, rows in files.items():
        text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
        (p / name).write_text(text)
        hashes[name] = hashlib.sha256(text.encode()).hexdigest()[:16]
    (p / "MANIFEST.json").write_text(json.dumps({"files": hashes, "reliability_values": RELIABILITY,
                                                  "note": "ASSIGNED TEST VALUES, not empirical"}, indent=2))
    return hashes


def load_controlled(d: str = "data/controlled"):
    p = Path(d)
    rd = lambda n: [json.loads(l) for l in (p / n).read_text().splitlines() if l.strip()]
    return rd("responses.jsonl"), rd("scenarios.jsonl")
