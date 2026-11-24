"""No-API tests: data invariants, judge caching/errors, adapter wiring & error paths."""

import json
import os
import stat
import textwrap

import pytest

from benchmark import data_factory as df
from benchmark.judges.base import Judge, JudgeError, JudgeResponse
from benchmark.judges.codex import CodexJudge


# --------------------------------------------------------------------- data
def test_five_main_responses_exact():
    main = [r for r in df.RESPONSES if r["main"]]
    assert [r["response_id"] for r in main] == ["R1", "R2", "R3", "R4", "R5"]
    assert main[0]["text"] == "Alex Meron was born on 3 March 1990."
    assert main[3]["text"] == "The exact date of birth is uncertain."


def test_scenarios_change_one_variable():
    sc = {s["scenario_id"]: s for s in df.build_scenarios()}
    c1 = sc["C1"]
    text = lambda s: [d["content"] for d in s["documents"]]
    # reliability-only scenarios: identical text, different metadata
    for sid in ("C2", "C3"):
        assert text(sc[sid]) == text(c1)
        assert [d["oracle_reliability"] for d in sc[sid]["documents"]] != [d["oracle_reliability"] for d in c1["documents"]]
    assert "probably" in sc["C4"]["documents"][1]["content"] and "probably" not in sc["C4"]["documents"][0]["content"]
    assert text(sc["ORD_BA"]) == list(reversed(text(c1)))
    dup = sc["DUP_A3"]["documents"]
    assert len({d["source_id"] for d in dup if d["doc_id"] == "A"}) == 1 and len(dup) == 4
    assert text(sc["C1_Q_YEAR"]) == text(c1) and sc["C1_Q_YEAR"]["question"] != c1["question"]
    assert sc["NC"]["documents"][1]["stance"]["day"] == 3


def test_no_gold_inferred():
    assert df.FIELD_REFERENCE["independent_gold"] is None
    assert df.FIELD_REFERENCE["day"]["status"] == "contested"


# -------------------------------------------------------------- codex judge
FAKE_CODEX = textwrap.dedent("""\
    #!/usr/bin/env python3
    import sys, json, os
    a = sys.argv; prompt = sys.stdin.read()
    if os.environ.get("FAKE_CODEX_FAIL"):
        print(json.dumps({"type": "error", "message": "You've hit your usage limit"})); sys.exit(1)
    out = a[a.index("-o") + 1]
    open(out, "w").write("ECHO:" + prompt[:20])
    print(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 2}}))
    """)


@pytest.fixture
def fake_codex(tmp_path):
    p = tmp_path / "codex"
    p.write_text(FAKE_CODEX)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


def test_codex_judge_cache_roundtrip(tmp_path, fake_codex):
    j = CodexJudge(codex_bin=fake_codex, cache_dir=str(tmp_path / "c"), call_log=str(tmp_path / "log.jsonl"))
    r1 = j.complete("hello world")
    r2 = j.complete("hello world")
    assert r1.text == "ECHO:hello world" and not r1.cache_hit and r2.cache_hit
    assert r1.usage["input_tokens"] == 10
    rec = json.loads(next((tmp_path / "c").iterdir()).read_text())
    assert rec["prompt"] == "hello world" and rec["model"] == j.model


def test_codex_judge_offline_miss_raises(tmp_path, fake_codex):
    j = CodexJudge(codex_bin=fake_codex, cache_dir=str(tmp_path / "c"), offline=True)
    with pytest.raises(JudgeError):
        j.complete("never cached")


def test_codex_judge_quota_error_not_retried(tmp_path, fake_codex, monkeypatch):
    monkeypatch.setenv("FAKE_CODEX_FAIL", "1")
    j = CodexJudge(codex_bin=fake_codex, cache_dir=str(tmp_path / "c"), call_log=str(tmp_path / "l.jsonl"))
    led = j.open_ledger()
    with pytest.raises(JudgeError, match="usage limit"):
        j.complete("x")
    assert led.n_errors == 1
    assert sum(1 for l in open(tmp_path / "l.jsonl") if '"error"' in l) == 1


def test_codex_judge_call_cap(tmp_path, fake_codex):
    j = CodexJudge(codex_bin=fake_codex, cache_dir=str(tmp_path / "c"), max_new_calls=1)
    j.complete("a")
    with pytest.raises(JudgeError, match="cap"):
        j.complete("b")


def test_codex_unavailable_binary(tmp_path):
    j = CodexJudge(codex_bin=None, cache_dir=str(tmp_path / "c"))
    j.codex_bin = None
    with pytest.raises(JudgeError):
        j.complete("x")


# ------------------------------------------------------------ scripted judge
class ScriptedJudge(Judge):
    """Rule-based stand-in used ONLY to test plumbing; never used for reported results."""
    backend = "scripted-test"
    model = "scripted"

    def __init__(self, fail=False):
        super().__init__()
        self.fail = fail

    def _complete(self, prompt, system=None, tag="", output_schema=None):
        if self.fail:
            raise JudgeError("scripted failure")
        text = "<rationale>r</rationale><answer>SUPPORTS</answer>"
        return JudgeResponse(text, self.model, self.backend, False, 0.0, {}, "t")


def _items():
    sc = {s["scenario_id"]: s for s in df.build_scenarios()}["C1"]
    ctx = [df.render_doc(d) for d in sc["documents"]]
    return [dict(item_id=r["response_id"], query=sc["question"], answer=r["text"], contexts=ctx, scenario_metadata={})
            for r in df.RESPONSES if r["response_id"] in ("R1", "R3")]


def test_ragchecker_error_path():
    from benchmark.evaluators.ragchecker import RAGCheckerAdapter
    res = RAGCheckerAdapter(ScriptedJudge(fail=True)).evaluate_many(_items()[:1], mode="SHARED_MONTH")
    assert res[0].status == "ERROR" and all(m.native_metric_value is None for m in res[0].metrics)


def test_ragchecker_no_reference_marks_not_applicable():
    from benchmark.evaluators.ragchecker import RAGCheckerAdapter
    res = RAGCheckerAdapter(ScriptedJudge(fail=True)).evaluate_many(_items()[:1], mode="NO_REFERENCE")
    na = [m for m in res[0].metrics if m.status == "NOT_APPLICABLE"]
    assert len(na) == 10 and all(m.native_metric_name != "faithfulness" for m in na)


def test_qcare_error_path():
    from benchmark.evaluators.qcare import QCAREAdapter
    res = QCAREAdapter(ScriptedJudge(fail=True)).evaluate_many(_items()[:1])
    assert res[0].status == "ERROR" and all(m.native_metric_value is None for m in res[0].metrics)


def test_codex_cache_dir_is_absolute(tmp_path, fake_codex, monkeypatch):
    monkeypatch.chdir(tmp_path)
    j = CodexJudge(codex_bin=fake_codex, cache_dir="rel_cache")
    j.complete("p")
    os.chdir("/")
    assert j.complete("p").cache_hit


def test_ragchecker_golden_reference_is_passed():
    from benchmark.evaluators.ragchecker import RAGCheckerAdapter
    inst = df.grid(["C1"], ["Q_DATE"])[0]
    items = [dict(item_id=f"C1|Q_DATE|{r['response_id']}", query=inst["question"], answer=r["text"],
                  contexts=[df.render_doc(d) for d in inst["documents"]], scenario_metadata={},
                  reference=inst["reference"]) for r in df.RESPONSES if r["main"]]
    res = RAGCheckerAdapter(ScriptedJudge(fail=True)).evaluate_many(items, mode="GOLDEN")
    assert {r.raw["gt_answer"] for r in res} == {df.GOLDEN_C1_DATE}  # identical across R1-R5
    assert all(r.mode == "GOLD:C1:Q_DATE" for r in res)


def test_references_fixed_per_scenario_question():
    for inst in df.grid(["C1", "C2", "C4", "NC"], ["Q_YEAR", "Q_DATE"]):
        assert inst["reference"]["text"]
    assert df.golden_reference("C1", "Q_YEAR")["text"] == "Alex Meron was born in 1990."
    assert df.golden_reference("ORD_BA", "Q_DATE")["text"] == df.GOLDEN_C1_DATE


FAKE_CLAUDE = textwrap.dedent("""\
    #!/usr/bin/env python3
    import sys, json
    a = sys.argv; prompt = sys.stdin.read()
    assert "--tools" in a and a[a.index("--tools") + 1] == ""
    assert "--setting-sources" in a and "--no-session-persistence" in a and "--bare" not in a
    sysprompt = a[a.index("--system-prompt") + 1]
    print(json.dumps({"result": "SYS=" + sysprompt[:10] + "|" + prompt[:10], "is_error": False,
                      "usage": {"input_tokens": 5, "output_tokens": 1},
                      "modelUsage": {"claude-sonnet-5-5": {"inputTokens": 5}}, "total_cost_usd": 0.0}))
    """)


def test_claude_judge_minimal_harness(tmp_path):
    from benchmark.judges.claude import ClaudeJudge
    p = tmp_path / "claude"
    p.write_text(FAKE_CLAUDE)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    j = ClaudeJudge(model="claude-sonnet-5-5", claude_bin=str(p), cache_dir=str(tmp_path / "c"),
                    call_log=str(tmp_path / "l.jsonl"))
    r = j.chat([{"role": "system", "content": "You are an expert evaluator"}, {"role": "user", "content": "judge this"}])
    assert r.text == "SYS=You are an|judge this"
    log = [json.loads(l) for l in open(tmp_path / "l.jsonl")]
    assert log[0]["model_reported_by_cli"] == "claude-sonnet-5-5"
    # different backend never shares cache entries with codex
    from benchmark.judges.codex import CodexJudge
    assert CodexJudge(model="claude-sonnet-5-5")._key("judge this", "You are an expert evaluator", None) != \
        j._key("judge this", "You are an expert evaluator", None)


def test_identical_concurrent_prompts_share_one_call(tmp_path, fake_codex):
    """Regression: parallel identical prompts must not each call the judge."""
    j = CodexJudge(codex_bin=fake_codex, cache_dir=str(tmp_path / "c"), call_log=str(tmp_path / "l.jsonl"), max_workers=5)
    rs = j.batch(["same prompt"] * 5)
    log = [json.loads(l) for l in open(tmp_path / "l.jsonl")]
    assert sum(e["event"] == "call" for e in log) == 1 and len({r.text for r in rs}) == 1


# ------------------------------------------------- concurrent error attribution
class _BarrierJudge(ScriptedJudge):
    """Holds every item's first call at a barrier so all items are in flight together; fails one answer."""

    def __init__(self, n_items, fail_marker):
        super().__init__()
        self.barrier = __import__("threading").Barrier(n_items, timeout=20)
        self.fail_marker = fail_marker
        self.seen = set()

    def _complete(self, prompt, system=None, tag="", output_schema=None):
        if "Break down the following report" in prompt:
            self.barrier.wait()  # all items have open ledgers at this point
            if self.fail_marker in prompt:
                raise JudgeError("scripted failure for one item")
        return super()._complete(prompt, system, tag, output_schema)


def test_concurrent_ledgers_record_only_their_own_calls():
    import threading
    n, barrier, ledgers = 4, threading.Barrier(4, timeout=20), {}

    class Failing(ScriptedJudge):
        def _complete(self, prompt, system=None, tag="", output_schema=None):
            if prompt == "p2":
                raise JudgeError("boom")
            return super()._complete(prompt, system, tag, output_schema)
    j = Failing()

    def work(i):
        led = j.open_ledger()
        try:
            barrier.wait()
            try:
                j.complete(f"p{i}")
            except JudgeError:
                pass
            barrier.wait()  # nobody closes its ledger before every call has been recorded
        finally:
            j.close_ledger(led)
        ledgers[i] = led

    ts = [threading.Thread(target=work, args=(i,)) for i in range(n)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert [ledgers[i].n_errors for i in range(n)] == [0, 0, 1, 0]
    assert all(len(ledgers[i].calls) == 1 for i in range(n))


def test_failed_call_marks_only_dependent_item_under_concurrency():
    """Regression: one failed judge call must not mark unrelated concurrent items as ERROR."""
    from benchmark.evaluators.conflictscore import ConflictScoreAdapter
    sc = {s["scenario_id"]: s for s in df.build_scenarios()}["C1"]
    ctx = [df.render_doc(d) for d in sc["documents"]]
    items = [dict(item_id=r["response_id"], query=sc["question"], answer=r["text"], contexts=ctx, scenario_metadata={})
             for r in df.RESPONSES if r["main"] and r["response_id"] != "R5"]
    judge = _BarrierJudge(len(items), fail_marker="uncertain")  # R4 only
    res = ConflictScoreAdapter(judge, max_workers=len(items)).evaluate_many(items, mode="published_cs")
    status = {it["item_id"]: r.status for it, r in zip(items, res)}
    assert status == {"R1": "OK", "R2": "OK", "R3": "OK", "R4": "ERROR"}


def test_ledger_sees_errors_inside_parallel_batch(tmp_path):
    """A ledger opened before CodexJudge.batch fans out to worker threads still records their calls."""
    class FailingCodex(CodexJudge):
        def _complete(self, prompt, system=None, tag="", output_schema=None):
            if prompt == "bad":
                raise JudgeError("boom")
            return JudgeResponse("ok", "m", "fake", False, 0.0, {}, "t")
    j = FailingCodex(codex_bin=None, cache_dir=str(tmp_path / "c"), max_workers=3)
    led = j.open_ledger()
    try:
        with pytest.raises(JudgeError):
            j.batch(["a", "bad", "b"])
    finally:
        j.close_ledger(led)
    assert led.n_errors == 1 and len(led.calls) == 3


FAKE_CLAUDE_REFUSAL = textwrap.dedent("""\
    #!/usr/bin/env python3
    import json, sys
    sys.stdin.read()
    msg = ("API Error: Sonnet 5.5's safeguards flagged this message (https://www.anthropic.com/legal/aup). "
           "This sometimes happens with safe, normal conversations. Claude Code can't respond to this message "
           "with Sonnet 5.5.\\n\\nTry rephrasing the request in a new session or change your model.\\n\\n"
           "Learn more: https://support.claude.com/en/articles/8106465\\n\\nDetails: `[reasoning_extraction]`"
           "\\n\\nRequest ID: req_TEST\\n\\nMessage ID: msg_TEST")
    print(json.dumps({"type": "result", "subtype": "success", "is_error": True, "stop_reason": "refusal",
                      "terminal_reason": "api_error", "safety_stops": 2, "api_error_status": None,
                      "result": msg, "usage": {"input_tokens": 4, "output_tokens": 0},
                      "modelUsage": {"claude-sonnet-5-5": {}}}))
    sys.exit(1)
    """)


def test_claude_refusal_error_keeps_provider_details(tmp_path):
    """Regression: a safeguard refusal must be logged with its category, stop reason and request id
    (the error text used to be cut at 300 characters, which hid '[reasoning_extraction]')."""
    from benchmark.judges.claude import ClaudeJudge
    p = tmp_path / "claude"
    p.write_text(FAKE_CLAUDE_REFUSAL)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    j = ClaudeJudge(model="claude-sonnet-5-5", claude_bin=str(p), cache_dir=str(tmp_path / "c"),
                    call_log=str(tmp_path / "l.jsonl"), max_retries=0)
    with pytest.raises(JudgeError) as e:
        j.complete("judge this")
    for s in ("[reasoning_extraction]", "stop_reason=refusal", "safety_stops=2", "Request ID: req_TEST"):
        assert s in str(e.value)
    log = [json.loads(l) for l in open(tmp_path / "l.jsonl")]
    assert log[-1]["event"] == "error" and "[reasoning_extraction]" in log[-1]["error"]
    assert not list((tmp_path / "c").glob("*.json"))  # refusals are never cached as judgments
