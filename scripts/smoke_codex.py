"""One-example smoke test of all three official evaluators with the Codex judge.

Usage: python scripts/smoke_codex.py [--model gpt-6-luna] [--effort low] [--response R1] [--scenario C1]
"""

import argparse
import json
import time
from pathlib import Path

from benchmark import data_factory as df
from benchmark.judges.codex import CodexJudge, codex_status

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gpt-6-luna")
ap.add_argument("--effort", default="low")
ap.add_argument("--response", default="R1")
ap.add_argument("--scenario", default="C1")
ap.add_argument("--rendering", default="attributed")
ap.add_argument("--out", default="outputs/smoke_codex")
ap.add_argument("--max-new-calls", type=int, default=40)
ap.add_argument("--evaluators", default="conflictscore_published,ragchecker,qcare")
a = ap.parse_args()

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
status = codex_status()
print("codex:", {k: v for k, v in status.items() if k != "codex_bin"})
judge = CodexJudge(model=a.model, reasoning_effort=a.effort, cache_dir="cache/judge",
                   call_log=str(out / "judge_calls.jsonl"), max_new_calls=a.max_new_calls)
sc = {s["scenario_id"]: s for s in df.build_scenarios()}[a.scenario]
resp = {r["response_id"]: r for r in df.RESPONSES}[a.response]
item = dict(item_id=f"{a.scenario}|{a.response}", query=sc["question"], answer=resp["text"],
            contexts=[df.render_doc(d, a.rendering) for d in sc["documents"]], scenario_metadata=sc)
print("question:", item["query"]); print("answer:", item["answer"]); print("contexts:", item["contexts"])

results = []
sel = a.evaluators.split(",")
if "conflictscore_published" in sel:  # paper Sec. 3 CS-C/CS-R (App. E, Tables 10-11)
    from benchmark.evaluators.conflictscore import ConflictScoreAdapter
    results += ConflictScoreAdapter(judge).evaluate_many([item], mode="published_cs")
if "ragchecker" in sel:
    from benchmark.evaluators.ragchecker import RAGCheckerAdapter
    results += RAGCheckerAdapter(judge).evaluate_many([item], mode="SHARED_MONTH")
if "qcare" in sel:
    from benchmark.evaluators.qcare import QCAREAdapter
    results += QCAREAdapter(judge).evaluate_many([item])

rows = [r.to_dict() for r in results]
for r in rows:
    r.pop("scenario_metadata", None)
(out / "results.json").write_text(json.dumps({"judge": judge.describe(), "item": {k: v for k, v in item.items() if k != "scenario_metadata"},
                                              "results": rows, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1, default=str))
print()
for r in results:
    print(f"== {r.evaluator_name} [{r.mode}] status={r.status} err={r.error}")
    for m in r.metrics:
        print(f"   {m.native_metric_name:32s} {m.native_metric_value!s:8s} ({m.status})")
    for w in r.warnings:
        print("   WARN", w)
calls = [json.loads(l) for l in open(out / "judge_calls.jsonl")]
new = [c for c in calls if c["event"] == "call"]
print(f"\njudge calls: {len(new)} new, {sum(c['event']=='cache_hit' for c in calls)} cache hits, "
      f"{sum(c['event']=='error' for c in calls)} errors; "
      f"input_tokens={sum(c['usage'].get('input_tokens',0) for c in new)} "
      f"cached_input={sum(c['usage'].get('cached_input_tokens',0) for c in new)} "
      f"output_tokens={sum(c['usage'].get('output_tokens',0) for c in new)}")
