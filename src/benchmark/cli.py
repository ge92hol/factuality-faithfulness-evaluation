"""CLI: python -m benchmark.cli {repos,data,run,analyze}."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from . import data_factory as df


def _portable_argv():
    """sys.argv with an absolute script path made relative to the working directory (no home paths in run_meta)."""
    argv = list(sys.argv)
    if argv and Path(argv[0]).is_absolute():
        try:
            argv[0] = str(Path(argv[0]).relative_to(Path.cwd()))
        except ValueError:
            argv[0] = Path(argv[0]).name
    return argv


EVALUATORS = {"conflictscore_published", "ragchecker", "qcare"}


def make_judge(spec: str, out: Path, max_new_calls: int | None, replicate: int = 0):
    """spec = backend:model[:effort], e.g. codex:gpt-6-luna:low"""
    parts = spec.split(":")
    backend = parts[0]
    if backend == "codex":
        from .judges.codex import CodexJudge
        return CodexJudge(model=parts[1], reasoning_effort=parts[2] if len(parts) > 2 else "low",
                          cache_dir="cache/judge", call_log=str(out / "judge_calls.jsonl"),
                          max_new_calls=max_new_calls, replicate=replicate)
    if backend == "claude":
        from .judges.claude import ClaudeJudge
        return ClaudeJudge(model=parts[1], cache_dir="cache/judge", call_log=str(out / "judge_calls.jsonl"),
                           max_new_calls=max_new_calls, replicate=replicate)
    raise SystemExit(f"unknown judge backend {backend}")


def cmd_repos(a):
    from .evaluators.base import git_sha
    from .judges.codex import codex_status
    for r in ["ConflictScore", "RAGChecker", "Q-CaRE-COLM-26", "RA-RAG", "llm-uncertainty-perceptions", "metametrics"]:
        print(f"{r:30s} {git_sha(r)}")
    print("codex:", {k: v for k, v in codex_status().items() if k != "codex_bin"})


def cmd_data(a):
    print(df.write_controlled(a.output))


def cmd_run(a):
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    judge = make_judge(a.judge, out, 0 if a.cache_only else a.max_new_calls, a.replicate)
    judge.offline = a.cache_only  # cache-only replay: a cache miss is an ERROR, never a judge call
    insts = df.grid(a.scenarios.split(","), a.questions.split(","))
    if a.reference_from:  # control: hold the RAGChecker reference fixed to another scenario's golden reference
        for inst in insts:
            inst["reference"] = df.golden_reference(a.reference_from, inst["question_id"])
    resp = [r for r in df.RESPONSES if (a.responses == "main" and r["main"]) or a.responses == "all"
            or r["response_id"] in a.responses.split(",")]
    items = []
    for inst in insts:
        ctx = [df.render_doc(d, a.rendering) for d in inst["documents"]]
        for r in resp:
            items.append(dict(item_id=f"{inst['instance_id']}|{r['response_id']}", query=inst["question"],
                              answer=r["text"], contexts=ctx, scenario_metadata=inst, reference=inst["reference"],
                              _inst=inst, _resp=r))
    ref_hash = hashlib.sha256(Path("data/controlled/references.jsonl").read_bytes()).hexdigest()[:16]
    meta = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "argv": _portable_argv(), "judge": judge.describe(),
            "rendering": a.rendering, "variant": a.variant, "reference_from": a.reference_from,
            "references_sha256_16": ref_hash, "n_items": len(items),
            "implementation_version": __import__("benchmark.evaluators.base", fromlist=["x"]).IMPLEMENTATION_VERSION,
            "context_path": "oracle (both scenario documents, scenario order)"}
    print(json.dumps(meta, indent=1))
    ev = a.evaluators.split(",")
    results = []

    def emit(res_list, its):
        with open(out / "results.jsonl", "a") as f:
            for r, it in zip(res_list, its):
                d = r.to_dict()
                d.update(item_id=it["item_id"], scenario=it["_inst"]["scenario_id"], question_id=it["_inst"]["question_id"],
                         question=it["query"], response_id=it["_resp"]["response_id"], response_label=it["_resp"]["label"],
                         response_text=it["answer"], contexts=it["contexts"], rendering=a.rendering, variant=a.variant,
                         documents=[{k: d[k] for k in ("doc_id", "source_name", "content", "reliability_label",
                                                       "oracle_reliability", "source_hedge")} for d in it["_inst"]["documents"]],
                         judge_spec=a.judge, reference_id=(it["reference"] or {}).get("reference_id"),
                         reference_text=(it["reference"] or {}).get("text"))
                f.write(json.dumps(d, default=str) + "\n")
                results.append(d)
                print(f"  {d['evaluator_name']:13s} {d['mode']:22s} {d['item_id']:22s} {d['status']}")

    if "conflictscore_published" in ev:  # paper Section 3 CS-C/CS-R
        from .evaluators.conflictscore import ConflictScoreAdapter
        emit(ConflictScoreAdapter(judge, max_workers=a.workers).evaluate_many(items, mode="published_cs"), items)
    if "ragchecker" in ev:
        from .evaluators.ragchecker import RAGCheckerAdapter
        rc = RAGCheckerAdapter(judge)
        for inst in insts:  # one official evaluate() per scenario x question
            its = [it for it in items if it["_inst"]["instance_id"] == inst["instance_id"]]
            emit(rc.evaluate_many(its, mode="GOLDEN" if inst["reference"] else "NO_REFERENCE"), its)
    if "qcare" in ev:
        from .evaluators.qcare import QCAREAdapter
        emit(QCAREAdapter(judge, max_workers=a.workers).evaluate_many(items), items)
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    meta["status_counts"] = {s: sum(r["status"] == s for r in results) for s in {r["status"] for r in results}}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta["status_counts"]))


def cmd_analyze(a):
    from .analysis import analyze
    analyze(a.inputs, a.report_dir)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="benchmark.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("repos"); s.add_argument("--check", action="store_true"); s.set_defaults(f=cmd_repos)
    s = sub.add_parser("data"); s.add_argument("--build-controlled", action="store_true")
    s.add_argument("--output", default="data/controlled/"); s.set_defaults(f=cmd_data)
    s = sub.add_parser("run")
    s.add_argument("--dataset", default="controlled")
    s.add_argument("--contexts", default="oracle", choices=["oracle"])
    s.add_argument("--scenarios", default="C1")
    s.add_argument("--questions", default="Q_YEAR,Q_DATE")
    s.add_argument("--responses", default="main")
    s.add_argument("--evaluators", default="conflictscore_published,ragchecker,qcare")
    s.add_argument("--rendering", default="attributed", choices=list(df.RENDERINGS))
    s.add_argument("--judge", default="codex:gpt-6-luna:low")
    s.add_argument("--replicate", type=int, default=0)
    s.add_argument("--variant", default="main", help="run label, e.g. main | ref_held_C1 | reliability_in_text")
    s.add_argument("--reference-from", default=None, help="use this scenario's golden reference (control)")
    s.add_argument("--max-new-calls", type=int, default=None)
    s.add_argument("--cache-only", action="store_true",
                   help="replay judgments from cache/judge only; never call the judge (cache miss -> ERROR)")
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--output", required=True); s.set_defaults(f=cmd_run)
    s = sub.add_parser("analyze"); s.add_argument("--inputs", nargs="+", required=True)
    s.add_argument("--report-dir", required=True); s.set_defaults(f=cmd_analyze)
    a = ap.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
