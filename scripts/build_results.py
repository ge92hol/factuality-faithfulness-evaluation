"""Build the canonical result tables in outputs/results/ from the runs in outputs/runs/.

Usage (repository root):  python scripts/build_results.py [--check]
  --check  build into a temporary directory and verify that every file in outputs/results/
           is reproduced byte for byte (no judge calls; reads results.jsonl and cache/judge/ only).

The runs themselves are produced by scripts/run_all.sh. The numeric tables of
outputs/reports/evaluation_results_brief.tex are filled from these tables (fill_brief).
"""

from __future__ import annotations

import argparse
import filecmp
import json
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from benchmark.evaluators.base import IMPLEMENTATION_VERSION  # noqa: E402
from benchmark.report_data import (FINAL_RUNS, comparison, deltas_vs_c1, export_judge_records,  # noqa: E402
                                   invariance_checks, judge_comparison, label_agreement, traces)

RES = ROOT / "outputs/results"
BRIEF = ROOT / "outputs/reports/evaluation_results_brief.tex"
MAIN = ["R1", "R2", "R3", "R4", "R5"]
SCEN = ["C1", "C2", "C3", "C4", "C5", "NC", "ORD_BA", "DUP_A3"]
CODEX = "codex:gpt-6-luna:low"


# ------------------------------------------------------------------ ConflictScore tables
def conflictscore_table(cmp: pd.DataFrame) -> pd.DataFrame:
    cs = cmp[(cmp.evaluator == "ConflictScore") & (cmp.variant == "main")]
    piv = cs.pivot_table(index=["scenario", "question_id", "response_id"], columns=["mode", "metric"],
                         values="value", aggfunc="first")
    cols = {
        ("published_cs", "cs_c"): "cs_c",
        ("published_cs", "cs_r"): "cs_r",
        ("published_cs", "n_claims"): "n_claims",
        ("published_cs", "n_conflicted_claims"): "n_conflicted_claims",
        ("published_cs", "n_claims_without_relevant_evidence"): "n_claims_excluded_from_cs_r",
    }
    piv = piv[[c for c in cols if c in piv.columns]]
    piv.columns = [cols[c] for c in piv.columns]
    piv = piv.reset_index()
    piv["scenario"] = pd.Categorical(piv["scenario"], SCEN, ordered=True)
    return piv.sort_values(["question_id", "scenario", "response_id"])


def ranking_string(series, higher_better):
    s = series.dropna()
    if s.empty:
        return ""
    groups = {}
    for rid, v in s.sort_values(ascending=not higher_better).items():
        groups.setdefault(round(float(v), 6), []).append(rid)
    return " > ".join("=".join(ids) + f"({v:.2f})" for v, ids in groups.items())


def conflictscore_rankings(cs: pd.DataFrame) -> pd.DataFrame:
    spec = [("cs_c", False), ("cs_r", False)]
    rec = []
    for (q, sc), g in cs.groupby(["question_id", "scenario"], observed=True):
        g = g.set_index("response_id")
        rec.append({"question_id": q, "scenario": sc,
                    **{f"ranking_{c}": ranking_string(g[c], hb) for c, hb in spec if c in g}})
    return pd.DataFrame(rec)


# ------------------------------------------------------------------ provenance of judgments
def judgment_provenance(runs):
    """Where every judgment used by a run comes from. All canonical runs are cache-only replays,
    so each judgment is a cache record created by an earlier judge call (timestamp, CLI version)."""
    rec = []
    for run in runs:
        meta = json.loads((ROOT / run / "run_meta.json").read_text())
        ev = [json.loads(l) for l in open(ROOT / run / "judge_calls.jsonl")]
        hits = {e["key"] for e in ev if e["event"] == "cache_hit"}
        miss = {e["key"] for e in ev if e["event"] == "error"}
        cache = [json.loads((ROOT / "cache/judge" / f"{k}.json").read_text()) for k in sorted(hits)]
        ts = sorted(c.get("timestamp", "") for c in cache)
        rec.append({
            "run": run, "judge": meta["argv"][meta["argv"].index("--judge") + 1],
            "cache_only_replay": bool(meta["judge"].get("cache_replay_only")),
            "new_judge_calls_in_this_run": sum(e["event"] == "call" for e in ev),
            "judgment_lookups": len(ev), "distinct_cached_judgments": len(hits),
            "judgments_generated_from": ts[0] if ts else "", "judgments_generated_to": ts[-1] if ts else "",
            "generating_cli_versions": "; ".join(f"{v} ({n})" for v, n in sorted(Counter(
                c.get("cli_version") or c.get("codex_version") or "unknown" for c in cache).items())),
            "served_model_reported": "; ".join(f"{v} ({n})" for v, n in sorted(Counter(
                str(c.get("model_reported_by_cli")) for c in cache).items())),
            "prompts_without_cached_judgment": len(miss),
            "results_ok": meta["status_counts"].get("OK", 0), "results_error": meta["status_counts"].get("ERROR", 0),
        })
    return pd.DataFrame(rec)


# ------------------------------------------------------------------ brief tables
def _f(v, signed=False):
    if v is None or pd.isna(v):
        return "--"
    v = round(float(v) + 0.0, 2) + 0.0
    return (f"{v:+.2f}" if v != 0 else "0.00") if signed else f"{v:.2f}"


def brief_blocks(cmp: pd.DataFrame, dl: pd.DataFrame) -> dict:
    main = cmp[(cmp.variant == "main") & (cmp.context_rendering == "attributed") & (cmp.judge == CODEX)]

    def val(sc, q, ev, mode, metric, r):
        g = main[(main.scenario == sc) & (main.question_id == q) & (main.evaluator == ev)
                 & (main["mode"].astype(str).str.startswith(mode)) & (main.metric == metric) & (main.response_id == r)]
        return g.value.iloc[0] if len(g) else None

    prim = [("ConflictScore", "published_cs", "cs_c"), ("ConflictScore", "published_cs", "cs_r"),
            ("RAGChecker", "GOLD", "faithfulness"), ("RAGChecker", "GOLD", "f1"),
            ("Q-CARE", "parsing", "verifiableness"), ("Q-CARE", "parsing", "completeness")]
    out = {}
    for q, tag in [("Q_DATE", "BASELINE_DATE"), ("Q_YEAR", "BASELINE_YEAR")]:
        out[tag] = "\n".join(f"{r} & " + " & ".join(_f(val("C1", q, *p, r)) for p in prim) + r" \\" for r in MAIN)
    ctrl = []
    for sc in ["NC", "ORD_BA", "DUP_A3"]:
        ctrl.append(f"{sc.replace('_', chr(92) + '_')} & " + " & ".join(
            f"{_f(val(sc, 'Q_DATE', *prim[0], r))} / {_f(val(sc, 'Q_DATE', *prim[1], r))}" for r in MAIN) + r" \\")
    out["CONTROLS"] = "\n".join(ctrl)
    # reliability: exact-date RC F1 deltas with scenario reference vs C1 reference held fixed
    rel = []
    for sc in ["C2", "C3"]:
        for variant, label in [("main", "Scenario-specific"), ("ref_held_C1", "C1 (held fixed)")]:
            g = dl[(dl.scenario == sc) & (dl.question_id == "Q_DATE") & (dl.evaluator == "RAGChecker")
                   & (dl.metric == "f1") & (dl.variant == variant) & (dl.context_rendering == "attributed")
                   & (dl.judge == CODEX)].set_index("response_id")
            rel.append(f"{sc} & {label} & " + " & ".join(_f(g.delta_vs_c1.get(r), True) for r in MAIN) + r" \\")
    out["RELIABILITY"] = "\n".join(rel)
    # hedges: every non-zero C4/C5 change in the six primary metrics
    names = {"cs_c": "CS-C", "cs_r": "CS-R", "faithfulness": "RC faithfulness", "f1": r"RC F1$^\dagger$",
             "verifiableness": "QC verifiableness", "completeness": "QC completeness"}
    hed = []
    for q, qn in [("Q_DATE", r"\textsc{Date}"), ("Q_YEAR", r"\textsc{Year}")]:
        first = True
        for sc in ["C4", "C5"]:
            for ev, mode, metric in prim:
                g = dl[(dl.scenario == sc) & (dl.question_id == q) & (dl.evaluator == ev) & (dl.metric == metric)
                       & (dl["mode"].astype(str).str.startswith(mode)) & (dl.variant == "main")
                       & (dl.context_rendering == "attributed") & (dl.judge == CODEX)].set_index("response_id")
                d = [g.delta_vs_c1.get(r) for r in MAIN]
                if all(x is None or pd.isna(x) or abs(x) < 1e-9 for x in d):
                    continue
                hed.append(f"{qn if first else ''} & {sc} / {names[metric]} & "
                           + " & ".join(_f(x, True) for x in d) + r" \\")
                first = False
        hed.append(r"\addlinespace[3pt]")
    out["HEDGES"] = "\n".join(hed[:-1])
    return out


def fill_brief(blocks: dict, path: Path) -> None:
    t = path.read_text()
    for k, body in blocks.items():
        pat = re.compile(rf"(% BEGIN {k}\n).*?(% END {k}\n)", re.S)
        if not pat.search(t):
            raise SystemExit(f"marker {k} missing in {path}")
        t = pat.sub(lambda m: m.group(1) + body + "\n" + m.group(2), t)
    path.write_text(t)


# ------------------------------------------------------------------ main
def build(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    runs = [str(ROOT / r) for r in FINAL_RUNS]
    cmp, rows = comparison(runs)
    cmp["implementation_version"] = IMPLEMENTATION_VERSION
    cmp.to_csv(out / "comparison.csv", index=False)
    dl = deltas_vs_c1(cmp)
    dl.to_csv(out / "deltas_vs_C1.csv", index=False)
    inv = invariance_checks(cmp)
    inv.to_csv(out / "invariance_checks.csv", index=False)
    cs = conflictscore_table(cmp)
    cs.to_csv(out / "conflictscore_published.csv", index=False)
    conflictscore_rankings(cs).to_csv(out / "conflictscore_rankings.csv", index=False)
    with open(out / "claim_level_traces.json", "w") as f:
        json.dump(traces(rows), f, indent=1)
    n_rec = export_judge_records(runs, out / "judge_records.jsonl")
    status = pd.DataFrame([{k: r.get(k) for k in ("item_id", "evaluator_name", "mode", "judge_spec", "variant",
                                                  "rendering", "status", "error")} for r in rows])
    status.to_csv(out / "run_status.csv", index=False)
    jw, ja = judge_comparison(cmp)
    jw.reset_index().to_csv(out / "judge_comparison_values.csv", index=False)
    ja.to_csv(out / "judge_comparison_agreement.csv", index=False)
    label_agreement(rows).to_csv(out / "judge_comparison_conflictscore_labels.csv", index=False)
    prov = judgment_provenance(FINAL_RUNS)
    prov.to_csv(out / "judgment_provenance.csv", index=False)
    meta = {
        "implementation_version": IMPLEMENTATION_VERSION,
        "runs": FINAL_RUNS,
        "run_commands": "scripts/run_all.sh (cache-only replay; no judge calls)",
        "n_results": len(rows),
        "status_counts": status.status.value_counts().to_dict(),
        "errors": [{k: e[k] for k in ("item_id", "evaluator_name", "mode", "judge_spec", "variant", "error")}
                   for e in status[status.status == "ERROR"].to_dict("records")],
        "n_judge_records": n_rec,
        "new_judge_calls": int(prov.new_judge_calls_in_this_run.sum()),
        "judgments": "every judgment is a cache record (cache/judge/) from an earlier judge call; see judgment_provenance.csv",
        "invariance_all_constant": bool(inv.constant_across_R1_R5.all()),
    }
    (out / "results_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    return {"meta": meta, "brief": brief_blocks(cmp, dl)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        with tempfile.TemporaryDirectory() as tmp:
            build(Path(tmp))
            names = sorted(p.name for p in RES.iterdir() if p.is_file() and p.name != "README.md")
            new = sorted(p.name for p in Path(tmp).iterdir())
            bad = [n for n in names if n not in new or not filecmp.cmp(RES / n, Path(tmp) / n, shallow=False)]
            bad += [n for n in new if n not in names]
            print(f"checked {len(names)} files: " + ("all identical" if not bad else f"DIFFER: {bad}"))
            sys.exit(1 if bad else 0)
    res = build(RES)
    fill_brief(res["brief"], BRIEF)
    print(json.dumps(res["meta"], indent=1))


if __name__ == "__main__":
    main()
