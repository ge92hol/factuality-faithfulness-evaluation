# Conflicting-Evidence RAG Evaluator Benchmark

## Claude Code implementation brief  \|  Version 1.0  \|  11 November 2026

*Sections 1–15 are the original specification; the file names there describe the planned layout. The implemented layout, commands and outputs are described in `README.md`.*

MISSION: Implement a reproducible, runnable comparison of existing factuality/faithfulness/RAG evaluators on the same questions, retrieved documents and candidate answers. Test whether published evaluators are sensitive to contradictions, source reliability, linguistic uncertainty and answer granularity. Produce actual results—not a literature-only review.

Scientific rule: DO NOT assume or try to ‘prove’ that every evaluator fails. Run controlled interventions, record real scores and identify which specific behaviors each implementation can and cannot detect. Distinguish metric limitations from retrieval failure, missing metadata, incorrect adaptation and unavailable software.

# 1\. Deliverables and definition of done

TODAY’S MINIMUM (MVP): a working Python repository; five fixed candidate answers on the birthday example; five evidence-condition scenarios; a real retrieval stage (BM25 minimum) with an oracle-context bypass; at least ConflictScore plus one other runnable evaluator; a CLI command; CSV/JSON outputs; a small Streamlit inspector if feasible; README, environment requirements, logs and limitations. Run real evaluator calls only if API credentials or local models are available; otherwise provide a testable adapter with explicit NOT\_RUN status—not fabricated metric outputs.

EXPANDED RESEARCH RUN: integrate RAGChecker and Q-CARE, evaluate an accessible subset of available benchmarks, compare ordinary versus reliability-aware retrieval, add source/answer hedging and controlled perturbations, produce charts, matched-case comparisons and a short error analysis.

Required artifacts: README.md; pyproject.toml or requirements.txt; .env.example (never secrets); configs/default.yaml; data/controlled/\*.jsonl; scripts/benchmark\_repos.py; src/benchmark/; tests/; outputs/results.csv; outputs/results.jsonl; outputs/report.md; outputs/coverage\_matrix.csv; optional app.py.

# 2\. What is being compared? Keep three experiments distinct

## A. Controlled evaluator comparison — PRIMARY

Give each evaluator exactly the same question, retrieved context and fixed candidate response. Compute its native metrics and compare sensitivities to controlled changes. The generation model is not being evaluated in this track. This isolates evaluator behavior.

## B. Real RAG generation comparison — SECONDARY

Index documents, retrieve them, feed them to one or more selected LLMs using a common prompt, and evaluate the generated answers. Compare standard relevance-based retrieval against a reliability-aware variant. Record model/prompt/temperature and retrieved document IDs. Do not confuse an LLM’s response-generation failure with an evaluator failure.

## C. Public benchmark replay — EXTERNAL VALIDATION

Use accessible examples from existing evaluation benchmarks and preserve their original annotations/protocols. Some benchmarks evaluate RAG systems rather than evaluators, and most lack explicit reliability/hedge labels. For those missing dimensions, construct a labeled synthetic overlay and report it separately from the unmodified benchmark results.

# 3\. Verified repositories and their roles

1\) ConflictScore — DIRECT EVALUATOR, first priority. [https\://github.com/microsoft/ConflictScore](https://github.com/microsoft/ConflictScore)  \| Paper: [https\://aclanthology.org/2026.emnlp-main.2589/](https://aclanthology.org/2026.emnlp-main.2589/) . Repository includes examples/conflict\_score\_example.py, claim–document SUPPORTS/CONTRADICTS/IRRELEVANT judgments and CS-C/CS-R. Inspect the example interface before adapting. IMPORTANT: the full ConflictBench data are not directly released in this repository; authors instruct users to reconstruct it from source datasets. Do not claim full benchmark reproduction until sourced successfully.

2\) RAGChecker — DIRECT EVALUATOR, second priority. [https\://github.com/amazon-science/RAGChecker](https://github.com/amazon-science/RAGChecker) . CLI/package supports query, response, gt\_answer and retrieved\_context. Some overall metrics require a ground-truth answer. A conflicting document set is NOT a single reliable GT; evaluate context-based metrics separately and, if needed, run explicitly labeled alternative-reference settings. Never quietly choose Source A’s date as truth.

3\) Q-CARE — DIRECT EVALUATOR plus existing benchmark. [https\://github.com/DISL-Lab/Q-CaRE-COLM-26](https://github.com/DISL-Lab/Q-CaRE-COLM-26)  \| Data: [https\://huggingface.co/datasets/DISLab/Q-CARE](https://huggingface.co/datasets/DISLab/Q-CARE) . Its Python example accepts query, answer and chunks; metrics include completeness, conciseness and verifiableness. It has 800 queries with retrieved chunks and answers. Default evaluator backbone requires substantial GPU memory (repository cites \~70GB bf16 for the Qwen3-30B configuration). Make this optional, with an alternative supported backbone and clear hardware note; do not promise CPU-friendly full reproduction.

4\) RA-RAG — SOURCE RELIABILITY / RETRIEVAL BASELINE, NOT a drop-in answer-quality evaluator. [https\://github.com/ml-postech/RA-RAG](https://github.com/ml-postech/RA-RAG)  \| Paper: [https\://aclanthology.org/2025.emnlp-main.1738/](https://aclanthology.org/2025.emnlp-main.1738/) . Read reliability estimation and retrieval/voting code; expose its estimated source-reliability signal if feasible. Keep oracle synthetic reliability and RA-RAG-estimated reliability distinct.

5\) Human interpretation of hedges — COMPONENT DATA/METHOD, NOT a full evaluator. [https\://github.com/UCIDataLab/llm-uncertainty-perceptions](https://github.com/UCIDataLab/llm-uncertainty-perceptions)  \| Paper: [https\://aclanthology.org/2024.emnlp-main.483/](https://aclanthology.org/2024.emnlp-main.483/) . Reuse available human-interpretation data for phrases such as ‘probably’/‘likely’. Inspect actual files and scales before implementing a phrase-to-commitment mapper; do not invent fixed paper-derived percentages.

6\) MetaMetrics — AGGREGATION BASELINE for a later supervised experiment, NOT required for today’s unsupervised evaluator benchmark. [https\://github.com/meta-metrics/metametrics](https://github.com/meta-metrics/metametrics)  \| Paper: [https\://proceedings.iclr.cc/paper\_files/paper/2025/hash/e1b248453bca182b6138b8c14a75340d-Abstract-Conference.html](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e1b248453bca182b6138b8c14a75340d-Abstract-Conference.html) . Supports metric configuration and optimizers, including XGBoost. Only train if independent preference targets exist; do not fit against labels calculated from the same final formula.

7\) GRANOLA — GRANULARITY REFERENCE, conditional implementation. [https\://aclanthology.org/2024.acl-long.365/](https://aclanthology.org/2024.acl-long.365/) . Accuracy and informativeness need multi-granularity reference answers. Implement an adapted, explicitly labeled test only after verifying the paper’s exact matching criterion and reference structure. Do not treat it as a plug-and-play conflicting-source RAG evaluator.

Other relevant external conflict benchmarks: ConfRAG (real-world conflicts, answer/reason coverage): [https\://aclanthology.org/2026.acl-long.11/](https://aclanthology.org/2026.acl-long.11/)  \| Dataset: [https\://huggingface.co/datasets/OracleY/ConfRAG](https://huggingface.co/datasets/OracleY/ConfRAG) ; ContextConflict (factual, temporal, granularity, perspective, ambiguity etc.): [https\://aclanthology.org/2026.emnlp-main.918/](https://aclanthology.org/2026.emnlp-main.918/) . Verify data/code access and record licensing before using. Avoid name confusion with the unrelated 2026 CONFRAG confidence-guided RAG paper.

# 4\. Core controlled test: five answers and five source conditions

## Canonical question and documents

Question: ‘When was the fictional person Alex Meron born?’ Use a fictional entity to avoid leakage from parametric knowledge.

Doc A (id=A, source=Archive A): ‘Alex Meron was born on 3 March 1990.’ Doc B (id=B, source=Registry B): ‘Alex Meron was born on 17 March 1990.’ Both sources agree on March 1990; they disagree on the day.

## Five fixed candidate responses (version-control this exact file)

R1 EXACT-A: ‘Alex Meron was born on 3 March 1990.’

R2 MONTH: ‘Alex Meron was born in March 1990.’

R3 YEAR: ‘Alex Meron was born in 1990.’

R4 VAGUE: ‘The exact date of birth is uncertain.’

R5 ATTRIBUTED-CONFLICT: ‘Archive A gives 3 March 1990, while Registry B gives 17 March 1990\. Both agree on March 1990, but the precise day is disputed.’

Additional response variants required: ‘probably 3 March 1990’; unqualified ‘17 March 1990’; unsupported ‘9 April 1992’; explicit false consensus ‘Both sources agree on 3 March 1990.’ Preserve the five main answers as the fixed comparison set.

## Five source-condition scenarios (illustrative controlled metadata)

C1 symmetric high reliability: A high \+ definite; B high \+ definite.

C2 asymmetric A: A high \+ definite; B low \+ definite.

C3 asymmetric B: A low \+ definite; B high \+ definite.

C4 source hedge B: A high \+ definite; B high \+ ‘probably born on 17 March 1990.’

C5 source hedge A: A high \+ ‘probably born on 3 March 1990’; B high \+ definite.

Implement an additional NO-CONFLICT control (same date in both documents) and retrieval-order reversal A/B and B/A. Store experimental reliability as explicit scenario labels and a numeric oracle field only for controlled tests (e.g., high=0.9/low=0.3 as ASSIGNED TEST VALUES, not empirically established reliability). Contrast source hedging with answer hedging. Keep phrasing variation limited so one manipulated variable changes at a time.

Minimum size: 5 answers × 5 conditions \= 25 fixed-answer evaluations per evaluator, plus controls and response variants. Cache repeated claim checks and report cost. If a full suite is too expensive, run a smaller prespecified smoke subset and label the remainder NOT\_RUN.

# 5\. Real RAG pipeline (required, but isolate retrieval from evaluation)

## Ingestion and indexing

Ingest JSONL documents and optional public benchmark passages. Each source must preserve doc\_id, source\_id, source name, URL/provenance, content, stance where known, timestamp, parent/source group, assigned/oracle reliability (synthetic only), estimated reliability (optional), and source hedge phrase. Chunk without losing original doc/source ID. Deduplicate near-identical passages by source group to avoid counting repeated copies as independent evidence.

Build a BM25 baseline. Optionally add dense retrieval (sentence-transformers \+ FAISS or Chroma) and reciprocal-rank fusion. Index only allowed benchmark train/test context appropriate to each scenario; no cross-question leakage. Retrieve top\_k with scores and document IDs; report if either conflicting document was missed.

## Two comparable retrieval pathways

PATH 1 — ORACLE / FIXED CONTEXT: bypass retrieval and supply both docs to every evaluator and generator. This is the primary fair test of metric sensitivity to conflict, reliability and hedge changes.

PATH 2 — RETRIEVED CONTEXT: use BM25 (and optionally dense/hybrid) to retrieve passages. Compare plain relevance ranking with an explicitly labeled reliability-aware ranking/retrieval option. Log relevance score, reliability value, ranking formula and source ID. Never covertly delete a credible dissenting source just because reliability-weighted top\_k ranks it lower.

Generation layer (secondary): one pluggable LLM (OpenAI-compatible API or local Transformers). Reuse a fixed prompt: ‘Answer the question using the provided documents, cite sources, distinguish agreement from disagreement, and express uncertainty when warranted.’ Temperature=0 if supported. Save prompts, retrieved contexts, model IDs and generated text. Optional additional models only after evaluators work.

# 6\. Where to put reliability, contradictions and hedges

## Data schema and separation of quantities

Document-level reliability: r\_i with source\_id, measurement method (‘oracle synthetic’, ‘RA-RAG estimated’, ‘external rating’), and provenance. It is NOT a direct positive answer-quality feature.

Source-level hedge interpretation: h\_i, the communicated commitment of a particular source statement, derived from a documented method/data. Answer-level commitment: a(c), separately extracted from each generated claim. These are not interchangeable.

Claim-to-document stance s\_ic ∈ {SUPPORTS, CONTRADICTS, IRRELEVANT}. Capture quoted spans; preserve attribution so ‘Source B claims 17 March’ is not treated as the answer endorsing 17 March.

## Provisional structured diagnostics (NOT final validated metric)

Candidate weight w\_i \= r\_i × h\_i; S(c) \= sum supporting w\_i; C(c) \= sum contradicting w\_i; relative support ratio p\_rel(c)=S/(S+C) only when denominator\>0. Illustrative commitment fit J(c)=1−\|a(c)−p\_rel(c)\|. Treat as provisional ablation features, not calibrated truth probabilities. Handle zero/weak evidence explicitly, dependent/duplicate sources, and mixed source hedges. Include variants without r, without h, and with r×h so we can isolate effects.

Other answer-level diagnostic aspects: (i) evidence fit/justified specificity, (ii) information retention of shared relevant facts, (iii) accurate explanation/attribution of conflicts, and (iv) question satisfaction. Do not double-count raw reliability or hedging as an independent ‘good answer’ reward. For today: output these as diagnostic flags/measurements; final optimized aggregation is a separate later research phase requiring independent labels.

# 7\. Unified adapters and exact data contracts

Use a common EvaluatorAdapter interface with evaluate(query, answer, contexts, scenario\_metadata, reference=None) \-\> EvaluationResult. Result fields: evaluator\_name, package/git\_commit, native\_metric\_name, native\_metric\_value, higher\_is\_better, native\_range, normalized\_for\_plot (optional), warnings, status {OK, NOT\_APPLICABLE, NOT\_RUN, ERROR}, runtime\_sec, token\_usage/cost, raw\_output\_path, evidence\_spans and config\_hash.

Do not convert diverse native metrics into a fake universal 0–1 quality score. Save each native score separately. For evaluations needing gt\_answer, allow REF\_A, REF\_B, SHARED\_MONTH and NO\_REFERENCE experimental modes and record which one was used.

For every adapter: read official README and example; pin repo SHA/version; create tiny smoke test; document required input fields, GPU/API constraints, and whether source reliability or hedge labels are actually consumed by the published implementation. A method that ignores an external metadata field is not automatically a failed method—it may be outside the paper’s scope.

# 8\. Public benchmark strategy

Benchmark B1: Q-CARE published testbed for evaluation baselines and compatibility checks; use a small, reproducible subset first and then optional full set. Since it ships retrieved chunks and model predictions, benchmark replay does not require rebuilding the original retrievers.

Benchmark B2: ConfRAG for naturally conflicting retrieved web references. Select a reproducible sample with explicit contradictions and enough source metadata, and document whether reliability/hedging metadata are truly present or artificially added.

Benchmark B3: ConflictBench only if source datasets are legally/accessibly obtained and reconstruction matches authors’ scripts. Otherwise use the public ConflictScore example on our controlled data; report ConflictBench as unavailable rather than claiming it ran.

Optional: ContextConflict for additional temporal/granularity conflicts; do not substitute LLM-generation metrics for evaluator evaluation. Maintain separate dataset ID, split, license, annotation source and preprocessing hash.

For benchmark replay, report both original published data and our controlled counterfactual interventions as distinct datasets. Never edit test examples in place or represent synthetic reliability assignments as real-world credibility ground truth.

# 9\. Experimental tests and analysis

T1 CONTRADICTION: With identical reliability, do metrics distinguish a contested exact day from a shared month? Does a response that attributes both sources score differently from one that falsely declares consensus?

T2 RELIABILITY: Change only r\_A/r\_B. Which evaluators change scores? Record whether the evaluator was designed to ingest reliability. A desirable direction for ranking must be prespecified, not inferred after seeing scores.

T3 SOURCE HEDGES: Replace only definite/‘probably’ wording in one source and check output change. Compare using published human-interpretation estimates rather than guessed hedge constants.

T4 ANSWER HEDGES: Change only answer commitment; retain same factual proposition. Check whether hedging is rewarded regardless of evidence or only when appropriate.

T5 SPECIFICITY / RETENTION: exact day vs month vs year vs ‘uncertain’. Test whether shared information is retained without overclaiming.

T6 ATTRIBUTION: compare ‘Source A claims X; Source B claims Y’ against a flat unqualified assertion. Inspect claim decomposition for quotation/attribution errors.

T7 RETRIEVAL: same evaluator on oracle fixed contexts vs BM25 retrieved top\_k; quantify whether failure arises because conflicting source was missed.

T8 ORDER / DUPLICATION: reverse passage order, repeat one source three times with the same source ID, change top\_k. Measure sensitivity to presentation and correlated evidence.

T9 REPLICATES: rerun stochastic LLM judges with controlled seeds where possible, report variance and judge model/prompt. Do not compare metrics as if they shared the same scale.

Report per metric: raw score; direction; rank order of the five answers; pairwise difference across each controlled intervention; invariant/sensitive/error; uncertainty or repeats; limitations. Generate outputs/coverage\_matrix.csv mapping each evaluator to: conflict detection, source reliability, source hedging, answer hedging, specificity, retention, conflict attribution, question satisfaction—with columns for ‘explicit native capability’, ‘empirically observed sensitivity’, ‘unknown’ and ‘not applicable’.

A result counts as a demonstrated limitation only if the evaluator was correctly configured, the task is within its stated scope, test data satisfy assumptions, and the observed behavior violates a prespecified target. Keep qualitative examples where metric scores are ambiguous; never claim every evaluator fails.

# 10\. CLI and lightweight app

CLI (implement exact entrypoints, or document equivalent tested syntax):

python \-m benchmark.cli repos \--check

python \-m benchmark.cli data \--build-controlled \--output data/controlled/

python \-m benchmark.cli index \--input data/controlled/ \--retriever bm25

python \-m benchmark.cli run \--dataset controlled \--contexts oracle \--evaluators conflictscore,ragchecker \--responses all \--output outputs/smoke/

python \-m benchmark.cli run \--dataset controlled \--contexts retrieved \--retriever bm25 \--evaluators conflictscore,ragchecker \--output outputs/retrieved/

python \-m benchmark.cli analyze \--inputs outputs/smoke/ outputs/retrieved/ \--report outputs/report.md

streamlit run app.py

Streamlit tabs: (1) Scenario editor: change A/B wording, r\_A/r\_B metadata, source hedges, answer hedge, date granularity. (2) Retrieval inspector: rank and full chunks, citations, source ID, reliability mode, flag if A or B missing. (3) Evaluator runner: select compatible adapters, enter five responses or generated response, run and track actual job status/errors/cost. (4) Results: native-score table, paired differences, color-neutral charts and export JSON/CSV. (5) Diagnostics: claim extraction/stance, attribution, metadata passed versus ignored.

UI must disclose when an evaluator cannot use a particular metadata field. No fabricated ‘overall score’ and no silent fallbacks. Cache API outputs with hashed prompts and store run IDs.

# 11\. Repository layout and implementation order

Suggested layout: src/benchmark/{schemas.py,data\_factory.py,ingest.py,retrieve.py,generate.py,hedges.py,reliability.py,claim\_stance.py,cli.py,evaluators/{base.py,conflictscore.py,ragchecker.py,qcare.py,granola.py},analysis.py}; data/controlled/; data/external/; configs/; tests/; app.py; outputs/.

Phase 0 — Inspect exact upstream code, licenses and hardware requirements; write an availability table with VERIFIED / BLOCKED / REQUIRES\_ADAPTATION. Record commit hashes.

Phase 1 — Build controlled dataset and schema with tests; verify five responses and five conditions; implement BM25 and oracle context modes.

Phase 2 — Integrate ConflictScore (official example first) and RAGChecker. Run reproducible smoke data; capture raw outputs and error logs.

Phase 3 — Implement Q-CARE if hardware permits; otherwise mark blocked and retain install/runner instructions. Add heuristic or paper-derived granularity diagnostics only with ‘adapted’ label.

Phase 4 — Add RA-RAG reliability estimator or an honest experimental oracle-reliability mode; add human-hedge mapping with provenance. Conduct controlled interventions and benchmark replay.

Phase 5 — Streamlit UI, exports, README, tests, plots, and final report. Stretch goal: MetaMetrics aggregation once independent preferences exist.

# 12\. Engineering safeguards and acceptance criteria

Must have deterministic synthetic fixture IDs; JSON schema validation; tests for duplicate source IDs, reversed order, missing hedge, zero supporting/contradicting evidence, unavailable model, no API key, and conflicting references. Preserve question-level group splits to prevent leakage. Include exact dependency versions and seed values.

Never expose API keys in code or artifacts. Use environment variables and safe logs. Avoid downloading huge models or datasets automatically without an explicit configuration opt-in. Keep a small cost cap and a dry-run mode. Check licenses before caching third-party datasets in repo.

Completion checklist: a new user can install environment, run a no-API smoke test with fixture data, run an actual evaluator with configured credentials/model, view the five-answer comparison, switch oracle/retrieved contexts, export raw outputs, and see which metrics were NOT\_RUN, NOT\_APPLICABLE or FAILED. Report must contain at least one concrete case-level comparison without claiming universal failure.

# 13\. Suggested Claude Code instruction (copy/paste)

‘Implement the specification in this document as a runnable Python research prototype. Begin by verifying the linked upstream repositories, interfaces, licenses and compute requirements. Do not hallucinate commands, metric values or benchmark availability. Prioritize a minimum end-to-end working pipeline today: fixed five-answer birthday tests, five source conditions, real BM25 retrieval \+ oracle-context mode, official ConflictScore and RAGChecker adapters, reproducible outputs, CLI, and simple Streamlit results inspector. Add Q-CARE and public benchmark replay if feasible. Preserve native metric scales; expose source reliability and hedge metadata separately; perform controlled counterfactual tests. Write tests, README, .env.example, an availability matrix and a scientifically cautious report. On failure, provide exact blocker, logs and reproducible partial result instead of fake output.’

# 14\. Research references and notes

ConflictScore: [https\://aclanthology.org/2026.emnlp-main.2589/](https://aclanthology.org/2026.emnlp-main.2589/) ; source: [https\://github.com/microsoft/ConflictScore](https://github.com/microsoft/ConflictScore)

RAGChecker: [https\://github.com/amazon-science/RAGChecker](https://github.com/amazon-science/RAGChecker) ; Q-CARE: [https\://github.com/DISL-Lab/Q-CaRE-COLM-26](https://github.com/DISL-Lab/Q-CaRE-COLM-26)

RA-RAG: [https\://aclanthology.org/2025.emnlp-main.1738/](https://aclanthology.org/2025.emnlp-main.1738/) ; hedging: [https\://aclanthology.org/2024.emnlp-main.483/](https://aclanthology.org/2024.emnlp-main.483/)

MetaMetrics: [https\://proceedings.iclr.cc/paper\_files/paper/2025/hash/e1b248453bca182b6138b8c14a75340d-Abstract-Conference.html](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e1b248453bca182b6138b8c14a75340d-Abstract-Conference.html)

ConfRAG: [https\://aclanthology.org/2026.acl-long.11/](https://aclanthology.org/2026.acl-long.11/) ; Q-CARE benchmark: [https\://huggingface.co/datasets/DISLab/Q-CARE](https://huggingface.co/datasets/DISLab/Q-CARE)

Status at document creation: repositories/dataset landing pages inspected publicly; no evaluator repos were cloned, installed or executed as part of writing this specification. Numerical outcomes remain unknown until the coding experiments run.

# 15\. Source-coverage addendum from Factuality & Faithfulness review (13-paper audit)

This addendum checks the coding brief against the uploaded 34-page review. The review covers 13 papers; ConflictScore, MetaMetrics, and external benchmarks are supplementary to that core 13-paper roster. The audit does NOT imply that each paper implements a directly comparable evaluator.  
**FULL COVERAGE / RESEARCH ROLE (mark source repo and exact implementation status after inspection)**:  
1\. GRANOLA (ACL 2024, 2024.acl-long.365): independent gold hierarchy for correct exact/month/year answers; transfer/adaptation, not a native conflicting-context evaluator.  
2\. RAGChecker (2024): native diagnostic for gold-claim precision/recall and retrieved-context faithfulness; preserve these as separate outputs.  
3\. Q-CARE (COLM 2026): native question-subquery completeness, answer conciseness and evidence verifiableness; a single supporting chunk need not mean all sources agree.  
4\. CORE (Findings of ACL 2025, 2025.findings-acl.1018): informative nonredundant claim selection before factual verification. Missing from initial executable priorities; add a conditional CORE claim-filter adapter or a separately labeled approximation after checking open-source code.  
5\. CLC / Conformal Linguistic Calibration (NeurIPS 2025): correctness–specificity calibration via sampled answers, nested generalized claims and labeled calibration. This is a generation/calibration procedure, NOT a free-standing scorer for our five answers. Add an optional transfer experiment only if its actual algorithm and calibration data can be implemented.  
6\. Whose Facts Win? (ACL 2026, 2026.acl-long.1357): SOURCE-LABEL SWAP experiment. Measure the effect of labeling identical conflicting claims with different source identities; not a normative reliability score. Add as controlled generator-probe, if token probabilities and model are available.  
7\. RA-RAG (EMNLP 2025, 2025.emnlp-main.1738): consensus-estimated source reliability and reliability-weighted answer voting. Use a reliability estimator / RAG system baseline, not a quality metric for the five written answers.  
8\. CONFACT (IJCAI 2025, proceedings/2025/1073): credibility-aware evidence reranking or source-background prompting in binary fact checking. Optional generation/Yes-No benchmark track; do not force its macro-F1 onto free-form date answers.  
9\. CONTEXTCONFLICT (2026, arXiv:2609.03148): conflict-native reasoning accuracy and (for summary tasks) Evidence Balance via frozen-scorer Shapley contributions and normalized Gini inequality; lower imbalance is better when sources are equally valid. Implement only on a task-aligned summary setting. Its balance measure is NOT conflict-explanation correctness and assumes equally valid evidence.  
10\. CONFLICTS / DRAGged into CONFLICTS (2025, arXiv:2506.08500): conflict types (none, complementary, divergent viewpoints, freshness, misinformation), grounding, gold-answer recall where available, and binary type-specific response adherence judged by an LLM. For the birth-date test, first establish a defensible type/rubric; otherwise call it NOT\_APPLICABLE.  
11\. Navigating the Grey Area (EMNLP 2023, 2023.emnlp-main.335): prompt or answer-prefix hedge/booster manipulations with gold QA accuracy. This is a generation intervention, not a source-hedge scoring module. Add as optional model-generation probe.  
12\. Perceptions of Linguistic Uncertainty (EMNLP 2024, 2024.emnlp-main.483): human distributions of communicated probability for phrases such as ‘probably’; a phrase-interpretation component, NOT a truth-rate or publisher-reliability metric.  
13\. Faithfully Express Intrinsic Uncertainty in Words (EMNLP 2024, 2024.emnlp-main.443): compare answer decisiveness with the generating model’s sampled-answer confidence (paper uses repeated generations). This assesses intrinsic model belief, not agreement among conflicting documents; conditional model-sampling study.  
**FULL-COVERAGE CONTRACT**: generate outputs/source\_coverage.csv with columns paper, URL, native\_task, required\_inputs, original\_metric, implementation/repo, runnable\_status, adaptation\_flag, tested\_on\_five\_answers, reason\_if\_not\_applicable. Account explicitly for ALL 13 papers; do not present blank or skipped papers as failed evaluators.  
**ESSENTIAL ADDITION**: TWO QUESTION INTENTS ON IDENTICAL EVIDENCE. Run five answers under BOTH ‘Which year was Alex Meron born?’ and ‘What exact date was Alex Meron born?’; keep the two date documents identical. Year-only can be fully sufficient for the first question and informationally incomplete for the second. Compare answer sufficiency separately from truth and source support.  
**FIELD-LEVEL REFERENCE**: record year=1990 (both sources), month=March (both), day=3 (A) versus day=17 (B). If an independent gold birthday exists, store it in another field; NEVER infer real truth from source reliability, majority count or a single supported passage. Flag contested\_day, shared\_month\_year, attribution\_accuracy and requested\_granularity.  
**INTERVENTION CONTROLS**: hold all else fixed and vary only one of source reliability, source hedging (‘probably’ versus definite), answer hedging, source order, duplicate-source repetition or gold truth. Distinguish a speaker’s communicated hedge from the generator’s internal confidence and from an estimated publisher reliability score.  
**TEST DESIGN AND INTERPRETATION**: The five-answer probe is a common stress test, not a universal published benchmark on which all 13 papers should output a numeric grade. Run native protocols first; tag every transferred/adapted computation as adapted. Evaluate whether compatible existing metrics or transparent combinations already discriminate the five answers; do not presuppose failure. If claiming preference alignment, compare against blinded independent human judgments; synthetic structural labels only validate prespecified properties.  
**ADDITIONAL PAPER LINKS**: CORE [https\://aclanthology.org/2025.findings-acl.1018/](https://aclanthology.org/2025.findings-acl.1018/) ; Whose Facts Win? [https\://aclanthology.org/2026.acl-long.1357/](https://aclanthology.org/2026.acl-long.1357/) ; CONFACT [https\://www\.ijcai.org/proceedings/2025/1073](https://www.ijcai.org/proceedings/2025/1073) ; CONFLICTS [https\://arxiv.org/abs/2506.08500](https://arxiv.org/abs/2506.08500) ; GREY [https\://aclanthology.org/2023.emnlp-main.335/](https://aclanthology.org/2023.emnlp-main.335/) ; intrinsic uncertainty [https\://aclanthology.org/2024.emnlp-main.443/](https://aclanthology.org/2024.emnlp-main.443/) ; CONTEXTCONFLICT [https\://arxiv.org/abs/2609.03148](https://arxiv.org/abs/2609.03148) .  
**PRIORITY ADJUSTMENT**: MVP remains ConflictScore \+ RAGChecker (plus Q-CARE if feasible), five fixed responses, two question wordings, oracle and BM25 contexts, and controlled reliability/hedge cases. The seven newly surfaced works are required in the coverage audit, with conditional adaptations and optional experiments—not seven extra mandatory full reproductions.  

## Golden Reference Answers for RAGChecker

## Experiment C1: Equally reliable conflicting sources

### Question
What exact date was Alex Meron born?

### Retrieved evidence
- Archive A: Alex Meron was born on 3 March 1990.
- Registry B: Alex Meron was born on 17 March 1990.
- Both sources have equal, high reliability.

### Reference answer (evidence-grounded)
The retrieved records disagree about Alex Meron's
exact birth date. Archive A reports 3 March 1990,
whereas Registry B reports 17 March 1990.
Both agree on March 1990. The available evidence
does not establish which day is correct.

### Candidate answers to evaluate
- R1: Alex Meron was born on 3 March 1990.
- R2: Alex Meron was born in March 1990.
- R3: Alex Meron was born in 1990.
- R4: The exact date of birth is uncertain.
- R5: Archive A gives 3 March 1990, while
  Registry B gives 17 March 1990.
  Both agree on March 1990, but the precise
  day is disputed.

### Evaluation
Run R1–R5 through ConflictScore, RAGChecker,
and Q-CARE using the same evidence.

Use the reference answer for metrics that
require one. Record native numerical scores
and rankings separately.

## Reference answers for all scenarios and question types (implemented; frozen before scoring, C4/C5 Q_DATE revised 23 November 2026)

Source of truth in code: `src/benchmark/data_factory.py` (`GOLDEN_REFERENCES`, `ALT_REFERENCES`); frozen copy: `data/controlled/references.jsonl` (hash recorded in `data/controlled/MANIFEST.json` and in every run's `run_meta.json`). References were written before any candidate was scored with them, except GOLD:C4:Q_DATE and GOLD:C5:Q_DATE, which were revised on 23 November 2026 after initial scoring (revision recorded in `data/controlled/MANIFEST.json`; all other references unchanged). References are identical for R1–R5 within a scenario × question. They are **evidence-grounded ideal responses, not independently verified ground truth** (no true birthday exists for the fictional entity; `independent_gold = null`). Only RAGChecker consumes them; ConflictScore and Q-CARE are reference-free.

Question wordings (fixed, used everywhere): Q_YEAR = "Which year was Alex Meron born?"; Q_DATE = "What exact date was Alex Meron born?". The C1 golden reference text above is used verbatim for Q_DATE.

Construction principles (P1–P7): P1 answer at the requested granularity using only evidence content; P2 state fields all sources share plainly; P3 state disagreement at the requested granularity with attribution and without choosing a side under equal assigned reliability; P4 under unequal ASSIGNED reliability name the favoured source, still report the dissenting one, do not declare the conflict resolved; P5 report a source hedge as that source's tentative statement; under equal assigned reliability identify the definite (unhedged) report as better supported, while still stating the conflicting tentative report (C4/C5; revised 23 November 2026); P6 disagreement below the requested granularity is immaterial (year question → "Alex Meron was born in 1990." in every scenario, since all sources say 1990); P7 order reversal and duplication do not change evidence content → same reference as C1.

| Scenario | Q_DATE reference (abridged) | Principles |
|---|---|---|
| C1, ORD_BA, DUP_A3 | golden text above (A: 3 March vs B: 17 March; both March 1990; day not established) | verbatim / P7 |
| C2 (B low) | as C1, naming Archive A as higher-reliability; "evidence favours 3 March 1990 but does not conclusively establish" | P2–P4 |
| C3 (A low) | mirror of C2 favouring 17 March 1990 | P2–P4 |
| C4 (B hedged) | A reports 3 March 1990, B "states that he was probably born on 17 March 1990"; both March 1990; "Given their equal assigned reliability, 3 March 1990 is better supported by the supplied evidence because Archive A reports it without qualification, although a conflicting tentative report remains." | P2, P5 |
| C5 (A hedged) | mirror of C4: B reports 17 March 1990, A "states that he was probably born on 3 March 1990"; 17 March 1990 is better supported because Registry B reports it without qualification, although a conflicting tentative report remains | P2, P5 |
| NC (no conflict) | "Alex Meron was born on 3 March 1990." | P1, P2 |
| all scenarios, Q_YEAR | "Alex Meron was born in 1990." | P1, P2, P6 |

Reference-sensitivity probes (C1 × Q_DATE only, reported separately, never as the main result): ALT_MONTH "Alex Meron was born in March 1990; the exact day is not established by the evidence." (defensible); ALT_REF_A / ALT_REF_B single-source references (probes, not evidence-grounded under equal reliability). A hidden "true date" for a synthetic factuality experiment is NOT defined; if added later it must be stored separately and never inferred from reliability or majority.

## Execution status (implementation 1.0.0)

All runs are **adapted reproductions with a substituted judge**: official evaluator code (prompts, parsers, scoring) unchanged; only the LLM transport replaced. Primary judge: Codex CLI under a ChatGPT login, requested model `gpt-6-luna` (served model not reported by the CLI), reasoning effort low, temperature not controllable; every judgment cached by content hash; identical concurrent prompts share one call (single-flight). Judgments were generated on 11, 15, 19 and 23 November 2026 (Codex CLI 0.167.0-alpha.4.1, 0.169.0-alpha.11.1 and 0.171.0-alpha.13; Claude Code 2.1.322 and 2.1.329); per-run provenance is in `outputs/results/judgment_provenance.csv`. Reports: `outputs/reports/`; all numbers: `outputs/results/comparison.csv`.

| Experiment | Status | Output |
|---|---|---|
| R1–R5 × C1–C5, NC, ORD_BA, DUP_A3 × {Q_YEAR, Q_DATE}; ConflictScore CS-C/CS-R (paper Sec. 3, Tables 10–11), RAGChecker (evidence-grounded references), Q-CARE | **done, 240/240 OK** | `outputs/runs/main/` |
| Control: RAGChecker with reference held at C1 (C2, C3) | **done, 20/20 OK; identical to C1** | `outputs/runs/ref_held_C1/` |
| Reliability in text: C1–C3 × Q_DATE, reliability label rendered into document text (adapted) | **done, 45/45 OK** | `outputs/runs/reliability_in_text/` |
| Judge robustness: Claude Sonnet `claude-sonnet-5-5` (C1 × both questions), Opus `claude-opus-5-5` (C1 × Q_DATE) via the Claude Code CLI | **done, 30/30 and 15/15 OK** | `outputs/runs/judge_claude-sonnet/`, `outputs/runs/judge_claude-opus/` |
| Answer variants V1–V4, alternative references, BM25 retrieved contexts, replicates (T9), Q-CARE testbed replay | pending | — |
| Next frameworks (RAGBench/TRACe, RAGAS, RAGEval, GRANOLA, FActScore + CORE, ARES) | plan only | — |

All runs above are replayed from the judgment cache (`scripts/run_all.sh`, no new judge calls); tables in `outputs/results/`, reports in `outputs/reports/` (see `README.md`).

Reliability conditions C2/C3 differ from C1 only in metadata in the default rendering; identical scores are expected by construction and are not evidence of metric failure. With the label rendered into the text, no reference-free score changed either.
