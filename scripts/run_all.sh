#!/bin/bash
# Canonical runs behind outputs/results/. Run from the repository root.
#
# Default (cache-only): every judgment is read from the content-addressed cache cache/judge/;
# no judge is called and a cache miss is recorded as ERROR. This reproduces outputs/runs/ exactly.
# FRESH=1 scripts/run_all.sh  would call the judges for uncached prompts (needs Codex / Claude CLI logins).
# The Codex judge uses $CODEX_BIN if set, else the first `codex` on PATH; point CODEX_BIN at the
# logged-in Codex CLI if PATH resolves to another installation.
set -u
PY=${PY:-python}   # override with PY=/path/to/python if needed
MODE=--cache-only; [ "${FRESH:-0}" = 1 ] && MODE=""
CODEX=codex:gpt-6-luna:low
EV=conflictscore_published,ragchecker,qcare   # ConflictScore = the paper's CS-C/CS-R (App. E, Tables 10-11)
run() { d=outputs/runs/$1; shift; rm -rf "$d"; mkdir -p "$d"; echo "=== $d $(date +%T)"
        PYTHONPATH=src $PY -m benchmark.cli run --responses main --output "$d" $MODE "$@" > "$d/run.log" 2>&1
        echo "exit=$? $(tail -1 "$d/run.log")"; }

run main                --judge $CODEX --scenarios C1,C2,C3,C4,C5,NC,ORD_BA,DUP_A3 --questions Q_YEAR,Q_DATE --evaluators $EV --variant main
run ref_held_C1         --judge $CODEX --scenarios C2,C3 --questions Q_YEAR,Q_DATE --evaluators ragchecker --variant ref_held_C1 --reference-from C1
run reliability_in_text --judge $CODEX --scenarios C1,C2,C3 --questions Q_DATE --evaluators $EV --rendering attributed_reliability --variant reliability_in_text
run judge_claude-sonnet --judge claude:claude-sonnet-5-5 --scenarios C1 --questions Q_YEAR,Q_DATE --evaluators $EV --variant judge_sonnet
run judge_claude-opus   --judge claude:claude-opus-5-5   --scenarios C1 --questions Q_DATE --evaluators $EV --variant judge_opus
echo RUNS_DONE
