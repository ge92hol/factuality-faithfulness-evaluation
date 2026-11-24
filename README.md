# Conflicting-evidence RAG evaluator benchmark

A controlled study of how three published RAG evaluators — **ConflictScore**, **RAGChecker** and **Q-CARE** —
score answers when the retrieved sources conflict, differ in stated reliability, or hedge. Five fixed, hand-written
answers are scored against synthetic evidence, so that any change in a score comes from one controlled change
to the inputs. The aim is to see what each evaluator's native metrics reward or penalise. It is not a
replication of any paper's reported numbers.

Implementation version **1.0.0** (`src/benchmark/evaluators/base.py`). Research specification:
`docs_documents/factuality-faithfulness-evaluation-spec.md`.

## Evaluators

| Evaluator | Paper | What it measures | Code run here |
|---|---|---|---|
| ConflictScore | Liu et al., EMNLP 2026 | CS-C (share of claims with both supporting and contradicting evidence) and CS-R (contradiction severity); lower is better; reference-free | Paper Sec. 3 with the Appendix E prompts (Tables 10–11). Our own implementation, because the official repository has no CS-C/CS-R code; only its label parser is ported |
| RAGChecker | Ru et al., NeurIPS 2024 D&B | Claim-level precision/recall/F1 against a reference, faithfulness, hallucination, noise sensitivity, context metrics | Official `ragchecker==0.1.9` / `refchecker==0.2.17`, unchanged |
| Q-CARE | Choi et al., COLM 2026 | Completeness, conciseness, verifiableness, C-Prec@10, C-nDCG@10; reference-free | Official pipeline (`parsing="paper"`), unchanged, with an API judge in place of the local backbone |

Each evaluator reports only its own native metrics. There is no combined 0–1 score.
`outputs/reports/evaluator_implementation_guide.pdf` documents the formulas, the code paths and every
difference from the papers.

## Experimental setup

- **Evidence.** A fictional person's birth date, given by two documents that differ only in the day
  (3 vs 17 March 1990). Two question wordings: `Q_YEAR` and `Q_DATE`.
- **Answers** (`data/controlled/responses.jsonl`): R1 exact date from source A, R2 month only, R3 year only,
  R4 "uncertain", R5 states the conflict and attributes each date to its source.
- **Scenarios** (`data/controlled/scenarios.jsonl`): C1 equal sources; C2/C3 reliability metadata; C4/C5 one
  source hedged with "probably"; controls NC (no conflict), ORD_BA (reversed source order) and DUP_A3
  (duplicated source). The run `reliability_in_text` writes the reliability labels into the source text.
- **References** for RAGChecker (`data/controlled/references.jsonl`) were written by the project and are grounded
  in the evidence. Any revision is recorded in `data/controlled/MANIFEST.json`.
- **Judge.** One substituted judge fills every LLM role of all three evaluators: the Codex CLI, *requested* model
  `gpt-6-luna`, low reasoning effort. The CLI does not report the served model, so it is unverified, and
  temperature cannot be set. A cross-judge check on C1 uses Claude Sonnet 5.5 and Opus 5.5 through the Claude
  Code CLI. All results are therefore *adapted reproductions with a substituted judge*. Judge errors are
  reported as `ERROR` and are never scored.

## Installation

Requires Python ≥ 3.11, Git, and about 3 GB of disk space for PyTorch, which `refchecker` pulls in. No GPU is needed.

```bash
git clone <this repository> && cd factuality-faithfulness-evaluation
conda create -n ffe python=3.11 -y && conda activate ffe   # or any Python 3.11 virtualenv
pip install -e ".[dev]"
python -m spacy download en_core_web_sm                     # sentence splitter used by refchecker
scripts/setup_external.sh                                   # clone upstream repos into external/ at pinned commits
```

`external/` is not committed. `setup_external.sh` clones ConflictScore, Q-CARE, RAGChecker, RA-RAG,
llm-uncertainty-perceptions and metametrics at the commits the results were produced with. The Q-CARE adapter
imports the Q-CARE repository at runtime, and the tests also use the ConflictScore repository.
`python -m benchmark.cli repos` prints the commit of each clone.

## Running

All commands are run from the repository root.

```bash
pytest tests                              # offline tests; no judge calls
scripts/run_all.sh                        # replay all five runs into outputs/runs/ from cache/judge/ (no judge calls)
python scripts/build_results.py           # rebuild outputs/results/ and the results brief's tables
python scripts/build_results.py --check   # verify outputs/results/ byte for byte
```

Every judge response is stored in the content-addressed cache `cache/judge/`, which is committed. Each cache key is
a SHA-256 over the harness, model, effort, system text, prompt, output schema and replicate. `run_all.sh` runs
with `--cache-only`: it makes no judge calls, and a cache miss is recorded as an error. Set `PY=/path/to/python`
if `python` is not the environment's interpreter.

`FRESH=1 scripts/run_all.sh` calls the judges for any prompt that is not cached. This needs a logged-in Codex CLI,
or the Claude Code CLI for the cross-judge runs. Set `CODEX_BIN` if the logged-in `codex` is not the first one on
`PATH`. Without temperature control, fresh labels can differ from the cached ones.
`scripts/smoke_codex.py` scores a single example with live judge calls, as a connectivity check.

## Results

Runs (`outputs/runs/`): `main` (C1–C5 and the three controls × both questions × R1–R5; 240 results),
`ref_held_C1` (RAGChecker on C2/C3 with the C1 reference; 20), `reliability_in_text` (45), and
`judge_claude-sonnet` / `judge_claude-opus` (C1 re-judged; 30 / 15). All results have status OK.

Main findings (details in the results brief):

- ConflictScore is the only evaluator that penalises R1's contested day. RAGChecker faithfulness and Q-CARE
  verifiableness give R1 1.00, because one supporting source is enough and the other source's contradiction
  is not subtracted.
- No reference-free metric rewards R5 for acknowledging the conflict. Only RAGChecker's reference-based F1
  favours R5, and that F1 measures agreement with the project-authored reference.
- ConflictScore never sees the question, so R2 and R3 score 0/0 even when the exact date is asked. Q-CARE
  completeness does distinguish the requested granularity.
- Across judges, CS-C/CS-R are unchanged. The reference-based metrics are the least stable.

## Reports and output files

| What | Path |
|---|---|
| Results brief | `outputs/reports/evaluation_results_brief.pdf` (LaTeX source alongside) |
| Implementation summary (2 pages) | `outputs/reports/implementation_summary.pdf` |
| Technical guide: formulas, code paths, deviations from the papers | `outputs/reports/evaluator_implementation_guide.pdf` |
| All metric values (long format) | `outputs/results/comparison.csv` |
| ConflictScore CS-C/CS-R per answer, and rankings | `outputs/results/conflictscore_published.csv`, `conflictscore_rankings.csv` |
| Changes vs C1, invariance checks | `outputs/results/deltas_vs_C1.csv`, `invariance_checks.csv` |
| Cross-judge comparison | `outputs/results/judge_comparison_*.csv` |
| Claim-level labels, full judge records | `outputs/results/claim_level_traces.json`, `judge_records.jsonl` |
| Judgment provenance (generation dates, CLI versions) | `outputs/results/judgment_provenance.csv` |
| Run status, summary metadata | `outputs/results/run_status.csv`, `results_meta.json` |
| Raw runs | `outputs/runs/<run>/{results.jsonl,judge_calls.jsonl,run_meta.json,run.log}` |
| SHA-256 of all outputs | `outputs/MANIFEST_sha256.txt` (`sha256sum -c outputs/MANIFEST_sha256.txt`) |

## Repository layout

```
src/benchmark/      package: CLI, data factory, evaluator adapters, judge backends, analysis
tests/              offline tests (published worked examples, parsers, invariants)
scripts/            run_all.sh, build_results.py, setup_external.sh, smoke_codex.py
data/controlled/    scenarios, answers, references, distractors, MANIFEST.json
cache/judge/        cached judge responses used for every reported result
outputs/            runs/, results/, reports/, MANIFEST_sha256.txt
docs_documents/     research specification
```

Third-party code and licenses (the ported ConflictScore label parser, MIT, Microsoft) are listed in
`THIRD_PARTY_NOTICES.md`.

## Limitations

These are controlled synthetic probes, run with adapted implementations and a substituted judge. Each label is a
single judge sample, so judge variance is not estimated. Identical prompts share one cached judgment. The served
Codex model is unverified. Q-CARE runs on an API judge, outside its documented local-model setup. There are no
human gold labels, and these results do not show that any method fails in general.
