"""Data products for outputs/results/: comparison.csv, deltas vs C1, invariance checks,
condensed claim-level traces, judge-record export. Built by scripts/build_results.py."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from . import data_factory as df
from .analysis import load, long_table, rank_table

FINAL_RUNS = [  # canonical runs; commands in scripts/run_all.sh
    "outputs/runs/main",
    "outputs/runs/ref_held_C1",
    "outputs/runs/reliability_in_text",
    "outputs/runs/judge_claude-sonnet",   # cross-judge check: separate judge, separate variant label
    "outputs/runs/judge_claude-opus",
]
MAIN_RESPONSES = ["R1", "R2", "R3", "R4", "R5"]
# metrics that cannot depend on the candidate answer (same question, documents, reference)
ANSWER_INDEPENDENT = {("RAGChecker", "claim_recall"), ("RAGChecker", "context_precision"),
                      ("Q-CARE", "precision_at_10"), ("Q-CARE", "ndcg_at_10")}
SHORT = {"ConflictScore": "CS", "RAGChecker": "RC", "Q-CARE": "QC"}
MODE_SHORT = {"published_cs": "pub", "parsing=paper": ""}


def col_label(ev, mode, metric):
    m = MODE_SHORT.get(mode, "gold" if str(mode).startswith("GOLD") else mode)
    return f"{SHORT[ev]}{(':' + m) if m else ''}:{metric}"


def comparison(runs=FINAL_RUNS) -> pd.DataFrame:
    rows = load([r for r in runs if Path(r).exists()])
    lt = rank_table(long_table(rows))
    lt = lt[lt.response_id.isin(MAIN_RESPONSES)].copy()
    lt["context_rendering"] = lt["rendering"]
    lt["column"] = [col_label(e, m, k) for e, m, k in zip(lt.evaluator, lt["mode"], lt.metric)]
    cols = ["evaluator", "mode", "metric", "column", "scenario", "question_id", "response_id", "value", "rank",
            "n_distinct_values", "status", "higher_is_better", "range", "reference_dependent", "evidence_based",
            "reference_id", "judge", "context_rendering", "variant", "definition", "item_id"]
    return lt[cols].sort_values(["variant", "context_rendering", "question_id", "scenario", "evaluator", "mode",
                                 "metric", "response_id"]), rows


def deltas_vs_c1(cmp: pd.DataFrame) -> pd.DataFrame:
    """value(scenario) - value(C1) for the same question/evaluator/metric/response/rendering.
    RAGChecker's mode is the reference id, so it is matched on evaluator+metric, and the
    reference used on each side is recorded (reference switch vs evidence change)."""
    base = cmp[(cmp.scenario == "C1")]
    key = ["evaluator", "metric", "question_id", "response_id", "context_rendering", "judge"]
    b = base[base.variant.isin(["main", "reliability_in_text"])][key + ["value", "reference_id"]].rename(
        columns={"value": "c1_value", "reference_id": "c1_reference_id"})
    m = cmp[cmp.scenario != "C1"].merge(b, on=key, how="left")
    m["delta_vs_c1"] = m["value"] - m["c1_value"]
    def ref_text(rid):
        if not isinstance(rid, str) or not rid.startswith("GOLD:"):
            return None
        _, sc, q = rid.split(":")
        return df.golden_reference(sc, q)["text"]
    # compare reference TEXT (ids differ per scenario even when the text is identical)
    m["reference_changed"] = (m["evaluator"] == "RAGChecker") & (m["reference_id"].map(ref_text) != m["c1_reference_id"].map(ref_text))
    # a change in a reference-INDEPENDENT metric cannot be caused by a reference switch
    m["change_attributable_to_reference"] = m["reference_changed"] & m["reference_dependent"].astype(bool)
    return m


def invariance_checks(cmp: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (ev, metric), g0 in cmp.groupby(["evaluator", "metric"]):
        if (ev, metric) not in ANSWER_INDEPENDENT:
            continue
        for k, g in g0.groupby(["scenario", "question_id", "variant", "context_rendering", "judge"]):
            vals = g[g.status == "OK"]["value"]
            out.append({"evaluator": ev, "metric": metric, "scenario": k[0], "question_id": k[1], "variant": k[2],
                        "rendering": k[3], "n": len(vals), "n_distinct": vals.nunique(),
                        "constant_across_R1_R5": vals.nunique() <= 1})
    return pd.DataFrame(out)


def traces(rows: List[dict]) -> List[Dict]:
    """Condensed claim-level judgments per result (full raw stays in results.jsonl)."""
    out = []
    for r in rows:
        if r["response_id"] not in MAIN_RESPONSES:
            continue
        raw, t = r["raw"], {"item_id": r["item_id"], "variant": r.get("variant", "main"), "rendering": r["rendering"],
                            "evaluator": r["evaluator_name"], "mode": r["mode"], "judge": r["judge_spec"],
                            "status": r["status"]}
        if r["evaluator_name"] == "ConflictScore" and r["mode"] == "published_cs":
            t.update(claims=raw.get("claims"), claim_doc_labels=raw.get("claim_doc_labels"), cs=raw.get("cs"))
        elif r["evaluator_name"] == "RAGChecker":
            t.update({k: raw.get(k) for k in ("gt_answer", "reference_id", "response_claims", "gt_answer_claims",
                                               "retrieved2response", "answer2response", "response2answer", "retrieved2answer")})
        elif r["evaluator_name"] == "Q-CARE":
            t.update({k: raw.get(k) for k in ("decomposed_query", "atomic_facts", "relevance_check")})
        out.append(t)
    return out


def export_judge_records(runs, dest: Path) -> int:
    keys = set()
    for r in runs:
        f = Path(r) / "judge_calls.jsonl"
        if f.exists():
            keys |= {json.loads(l).get("key") for l in open(f)} - {None}
    n = 0
    with open(dest, "w") as f:
        for k in sorted(keys):
            p = Path("cache/judge") / f"{k}.json"
            if p.exists():
                f.write(json.dumps(json.loads(p.read_text())) + "\n")
                n += 1
    return n


PHASE_D_RUNS = ["outputs/runs/judge_claude-sonnet", "outputs/runs/judge_claude-opus"]


def judge_comparison(cmp: pd.DataFrame):
    """Same C1 inputs, different judges. Returns (wide value table, per-metric agreement summary)."""
    g = cmp[(cmp.scenario == "C1") & (cmp.context_rendering == "attributed")
            & cmp.variant.isin(["main", "judge_sonnet", "judge_opus"])
            & ~cmp.metric.isin(["n_supports", "n_contradicts", "n_irrelevant"])]
    if g.judge.nunique() < 2:
        return None, None
    w = g.pivot_table(index=["question_id", "column", "response_id"], columns="judge", values="value", aggfunc="first")
    base = "codex:gpt-6-luna:low"
    rows = []
    for (q, col), gg in w.groupby(level=[0, 1]):
        for j in [c for c in w.columns if c != base]:
            pair = gg[[base, j]].dropna()
            if pair.empty:
                continue
            same = (pair[base].round(6) == pair[j].round(6))
            rk = pair.rank(method="min", ascending=False)
            rows.append({"question_id": q, "metric": col, "judge": j, "n_answers": len(pair),
                         "identical_values": int(same.sum()), "max_abs_diff": float((pair[base] - pair[j]).abs().max()),
                         "same_ranking": bool((rk[base] == rk[j]).all()),
                         "codex_values": " ".join(f"{v:.2f}" for v in pair[base]),
                         "other_values": " ".join(f"{v:.2f}" for v in pair[j])})
    return w, pd.DataFrame(rows)


def label_agreement(rows: List[dict]) -> pd.DataFrame:
    """ConflictScore (CS-C/CS-R) claims and per-document labels, per judge (C1): one row per claim."""
    out = []
    for r in rows:
        if r["scenario"] != "C1" or r["evaluator_name"] != "ConflictScore" or r["mode"] != "published_cs" \
                or r.get("rendering") != "attributed" or r.get("variant") not in ("main", "judge_sonnet", "judge_opus"):
            continue
        for i, (claim, labels) in enumerate(zip(r["raw"].get("claims") or [], r["raw"].get("claim_doc_labels") or [])):
            out.append({"question_id": r["question_id"], "response_id": r["response_id"], "judge": r["judge_spec"],
                        "status": r["status"], "claim_index": i + 1, "claim": claim, "doc_labels": "/".join(labels)})
    return pd.DataFrame(out)
