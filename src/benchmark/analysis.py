"""Turn results.jsonl into long/wide tables and per-metric rankings (native scales only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pandas as pd

from .metric_defs import DEFS


def load(inputs: List[str]) -> List[dict]:
    rows = []
    for d in inputs:
        p = Path(d)
        for f in ([p] if p.is_file() else sorted(p.rglob("results.jsonl"))):
            rows += [json.loads(l) for l in open(f) if l.strip()]
    return rows


def long_table(rows: List[dict]) -> pd.DataFrame:
    recs = []
    for r in rows:
        for m in r["metrics"]:
            d = DEFS.get((r["evaluator_name"], m["native_metric_name"]), ("", None, "", None, None))
            recs.append({
                "scenario": r["scenario"], "question_id": r["question_id"], "response_id": r["response_id"],
                "evaluator": r["evaluator_name"], "mode": r["mode"], "judge": r["judge_spec"],
                "rendering": r["rendering"], "variant": r.get("variant", "main"), "metric": m["native_metric_name"], "value": m["native_metric_value"],
                "status": m["status"], "higher_is_better": d[1], "range": d[2], "reference_dependent": d[3],
                "evidence_based": d[4], "reference_id": r.get("reference_id") if r["evaluator_name"] == "RAGChecker" else None,
                "definition": d[0], "item_id": r["item_id"],
            })
    return pd.DataFrame(recs)


def rank_table(lt: pd.DataFrame) -> pd.DataFrame:
    """Rank R1-R5 within each (scenario, question, evaluator, mode, judge, rendering, metric).
    Rank 1 = best per native direction; ties share the best rank ('min'). No rank if direction undefined."""
    out = []
    keys = ["scenario", "question_id", "evaluator", "mode", "judge", "rendering", "variant", "metric"]
    for k, g in lt.groupby(keys, dropna=False):
        g = g.copy()
        hib = g["higher_is_better"].iloc[0]
        ok = g["status"] == "OK"
        if hib is None or pd.isna(hib) or not ok.any():
            g["rank"] = None
        else:
            g.loc[ok, "rank"] = g.loc[ok, "value"].astype(float).rank(ascending=not bool(hib), method="min")
        g["n_distinct_values"] = g.loc[ok, "value"].nunique()
        out.append(g)
    return pd.concat(out) if out else lt


def wide_tables(lt: pd.DataFrame):
    keys = ["scenario", "question_id", "evaluator", "mode", "judge", "rendering"]
    for k, g in lt.groupby(keys, dropna=False):
        w = g.pivot_table(index="response_id", columns="metric", values="value", aggfunc="first")
        w = w[[m for m in dict.fromkeys(g["metric"]) if m in w.columns]]
        yield dict(zip(keys, k)), w


def analyze(inputs: List[str], report_dir: str) -> None:
    out = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = load(inputs)
    lt = rank_table(long_table(rows))
    lt.to_csv(out / "metrics_long.csv", index=False)
    md = []
    for key, w in wide_tables(lt):
        md.append(f"### {key['evaluator']} [{key['mode']}] - {key['scenario']} / {key['question_id']} - judge {key['judge']}\n")
        md.append(w.to_markdown(floatfmt=".3f") + "\n")
        sub = lt[(lt.scenario == key["scenario"]) & (lt.question_id == key["question_id"]) & (lt.evaluator == key["evaluator"])
                 & (lt["mode"] == key["mode"]) & (lt.judge == key["judge"]) & lt["rank"].notna()]
        if len(sub):
            rk = sub.pivot_table(index="metric", columns="response_id", values="rank", aggfunc="first")
            md.append("Ranks (1 = best by native direction; ties share rank):\n\n" + rk.to_markdown(floatfmt=".0f") + "\n")
    (out / "tables.md").write_text("\n".join(md))
    defs = pd.DataFrame([{"evaluator": e, "metric": m, "definition": d[0], "higher_is_better": d[1], "range": d[2],
                          "reference_dependent": d[3], "evidence_based": d[4]} for (e, m), d in DEFS.items()])
    defs.to_csv(out / "metric_definitions.csv", index=False)
    status = pd.DataFrame([{k: r[k] for k in ("item_id", "evaluator_name", "mode", "judge_spec", "status", "error")} for r in rows])
    status.to_csv(out / "run_status.csv", index=False)
    print(f"wrote {out}/metrics_long.csv, tables.md, metric_definitions.csv, run_status.csv ({len(rows)} results)")
