"""Published ConflictScore response metric (CS-C, CS-R), as defined in the paper.

Source: Liu, Halfaker, Roth, Xia. "ConflictScore: Identifying and Measuring How
Language Models Handle Conflicting Evidence." EMNLP 2026, pp. 58825-58845.
https://aclanthology.org/2026.emnlp-main.2589/

Why this module exists
----------------------
The official repository (microsoft/ConflictScore @ a8868e5, == upstream HEAD on
2026-11-15) does NOT contain code for the paper's response-level metrics CS-C and
CS-R. Its `src/` pipeline implements the ConflictBench *conflict-detection*
protocol (one gold-derived claim per item), and its `examples/` script uses
different prompts and a support ratio S/(S+NS). This module implements the
paper's Section 3 procedure from the paper text itself:

  1. Claim decomposition with the paper's Table 10 prompt.
  2. Evidence evaluation of every (claim, document) pair with the paper's
     Table 11 prompt (labels SUPPORTS / CONTRADICTS / IRRELEVANT).
  3. Aggregation (paper Section 3, "ConflictScore Calculation"):
       conflicted(c) = |D+(c)| > 0 and |D-(c)| > 0
       CS-C = (1/n) * sum_c conflicted(c)                       (lower is better)
       CS-R = mean over claims of |D-(c)| / (|D+(c)| + |D-(c)|)  (lower is better)

Provenance of each component
----------------------------
ORIGINAL (paper text):  both prompts, the label set, the CS-C/CS-R formulas,
                        the conflict definition.
ORIGINAL (repo code):   `parse_label` reproduces, line for line, the label
                        parsing in eval_all_calibration.py lines 104-140
                        (JSON object -> "answer" field, regex fallback,
                        <answer> tag fallback, default IRRELEVANT).
ADAPTATION (ours):
  * Prompt transcription from the PDF: typographic quotes -> ASCII quotes,
    words split by line-break hyphenation rejoined ("infor-mation",
    "SUP-PORTS"), small-caps spacing "IRRELEV ANT" -> "IRRELEVANT",
    ". . ." -> the ellipsis character used by the repository prompt.
  * Claim-list parsing: the paper prints the instruction 'starting with
    "Claims: "' but publishes no parser. We accept both ways judges follow it:
    "Claims:" (or "Claim:") before every claim, or one "Claims:" header line
    followed by one claim per line. If nothing parses, the whole answer is
    used as one claim and a warning is recorded.
  * CS-R for a claim with |D+| + |D-| = 0 is undefined in the paper. Such
    claims are EXCLUDED from the CS-R mean (and counted in
    `n_claims_without_relevant_evidence`); they still count in the CS-C
    denominator as non-conflicted, as in the paper's Figure 2 example. If no
    claim has relevant evidence, CS-R is reported as None.
  * Judge errors are NOT converted to IRRELEVANT (the repository does that);
    an item with any judge error is reported with status ERROR.
  * The judge is substituted (paper main results: Gemini-3.5-Flash).
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Sequence

# --------------------------------------------------------------- prompts
# Paper Table 10 ("Prompt used for claim decomposition in ConflictScore").
DECOMPOSITION_PROMPT = """Instruction: Break down the following report into individual claims.

Report: {report}

Please identify each separate claim made in the report. A claim is a factual statement or opinion that expresses a belief or judgment.
List each claim on a new line, starting with "Claims: "."""

# Paper Table 11 ("Prompt used for evidence evaluation in ConflictScore").
EVIDENCE_PROMPT = """# Document-Claim Conflict Detection
You are a careful fact-checking assistant specializing in identifying conflicts between claims and evidence in documents.

Decide one of the following labels for how the document relates to the claim:
1. SUPPORTS – The document provides evidence that directly supports any part of the claim. This includes partial support: if a claim contains hedged or multi-part statements (e.g., "Coffee consumption can improve alertness but may also cause sleep disruption"), then documents that support any part of that claim should be labeled SUPPORTS.
2. CONTRADICTS – The document provides evidence that contradicts the claim (e.g., assigns an incompatible role/date/quantity/polarity, or states the opposite of an asserted relationship), even if it does not explicitly say the claim is false.
3. IRRELEVANT – The document does not provide information about the substantive content of the claim (mentions entities without giving information that could support or contradict) OR provides only vague/contextual information insufficient to judge the claim.

Decision rules:
- Look for statements in the document that address the same attributes asserted (who/what/when/where/how many/etc.) inside the claim.
- If the document contains a statement that supports the claim or partially supports any part of a hedged claim, choose SUPPORTS.
- If the document contains a statement incompatible with the claim, choose CONTRADICTS (do NOT choose IRRELEVANT).
- If it contains neither a compatible nor incompatible statement about the claim's asserted attributes, choose IRRELEVANT.

Example (contradiction via incompatible attribute):
- Claim: "Frédéric Chopin was a famous musician."
- Document: "Polish scientists born in Warsaw include … Frédéric Chopin."
- Label: CONTRADICTS (profession mismatch: scientist vs musician).

Now evaluate the following:

# Claim to evaluate
{claim}

# Document to evaluate against
{document}

# Output format
Return ONLY a single JSON object with these fields:
{{"claim": "{claim}",
"document_snippet": "<the most relevant snippet from the document that relates to the claim>",
"reasoning": "<a brief justification that explains your decision>",
"answer": "<SUPPORTS|CONTRADICTS|IRRELEVANT>"}}

Please be precise and follow the decision rules. Do not include any additional text or explanations and only output in the JSON format specified above."""

LABELS = ("SUPPORTS", "CONTRADICTS", "IRRELEVANT")


def decomposition_prompt(report: str) -> str:
    return DECOMPOSITION_PROMPT.format(report=report)


def evidence_prompt(claim: str, document: str) -> str:
    return EVIDENCE_PROMPT.format(claim=claim, document=document)


# --------------------------------------------------------------- parsing
_CLAIM_LINE = re.compile(r"^\s*(?:[-*•]\s*|\d+[.)]\s*)?Claims?\s*:\s*(.*\S)\s*$")
_CLAIM_HEADER = re.compile(r"^\s*(?:[-*•]\s*|\d+[.)]\s*)?Claims?\s*:\s*$")
_LIST_MARK = re.compile(r"^\s*(?:[-*•]\s*|\d+[.)]\s*)?(.*\S)\s*$")


def parse_claims(text: str) -> List[str]:
    """Claims from a Table 10 response. Table 10 asks to list each claim on a new line 'starting with
    "Claims: "', which judges follow in two ways: a prefix on every line ("Claims: X"; also "Claim: X",
    the repository example's convention) or one "Claims:" header followed by one claim per line.
    Both are accepted; in the header form a blank line ends the list."""
    out, in_list = [], False
    for line in (text or "").splitlines():
        m = _CLAIM_LINE.match(line)
        if m:
            out.append(m.group(1).strip())
            in_list = False
        elif _CLAIM_HEADER.match(line):
            in_list = True
        elif in_list:
            if not line.strip():
                in_list = False
                continue
            out.append(_LIST_MARK.match(line).group(1).strip())
    return out


# --- Adapted from microsoft/ConflictScore (commit a8868e5),
# src/eval_with_conflict_score/eval_all_calibration.py, lines 104-140.
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License; the full license text is in THIRD_PARTY_NOTICES.md.
def parse_label(evaluation: str) -> str:
    """Verbatim port of the label parsing in the official
    src/eval_with_conflict_score/eval_all_calibration.py (lines 104-140)."""
    result = "IRRELEVANT"
    try:
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', evaluation, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            parsed_json = json.loads(json_str)
            if "answer" in parsed_json:
                result = parsed_json["answer"].strip().upper()
            elif "support" in parsed_json:
                result = parsed_json["support"].strip().upper()
    except (json.JSONDecodeError, KeyError, AttributeError):
        try:
            json_match = re.search(r'"answer":\s*"([^"]+)"', evaluation, re.DOTALL | re.IGNORECASE)
            if json_match:
                result = json_match.group(1).strip().upper()
            else:
                json_match = re.search(r'"support":\s*"([^"]+)"', evaluation, re.DOTALL | re.IGNORECASE)
                if json_match:
                    result = json_match.group(1).strip().upper()
        except Exception:  # noqa: BLE001  (official code: bare except)
            pass
    if result == "IRRELEVANT":
        try:
            match = re.search(r"<answer>(.*?)</answer>", evaluation, re.DOTALL)
            if match:
                result = match.group(1).strip().upper()
        except Exception:  # noqa: BLE001
            pass
    return result


# ----------------------------------------------------------- aggregation
def claim_counts(labels: Sequence[str]) -> Dict[str, int]:
    return {"support": sum(l == "SUPPORTS" for l in labels),
            "contradict": sum(l == "CONTRADICTS" for l in labels),
            "irrelevant": sum(l not in ("SUPPORTS", "CONTRADICTS") for l in labels)}


def is_conflicted(labels: Sequence[str]) -> bool:
    """Paper Section 3: a claim has conflicts iff D+ and D- are both non-empty."""
    c = claim_counts(labels)
    return c["support"] > 0 and c["contradict"] > 0


def contradiction_ratio(labels: Sequence[str]) -> Optional[float]:
    """|D-| / (|D+| + |D-|); None when the claim has no supporting or contradicting document."""
    c = claim_counts(labels)
    den = c["support"] + c["contradict"]
    return None if den == 0 else c["contradict"] / den


def cs_metrics(claim_labels: Sequence[Sequence[str]]) -> Dict[str, Optional[float]]:
    """CS-C and CS-R for one response. `claim_labels[i]` = labels of claim i over all documents."""
    n = len(claim_labels)
    if n == 0:
        return {"cs_c": None, "cs_r": None, "n_claims": 0, "n_conflicted_claims": 0,
                "n_claims_without_relevant_evidence": 0}
    conflicted = [is_conflicted(l) for l in claim_labels]
    ratios = [contradiction_ratio(l) for l in claim_labels]
    defined = [r for r in ratios if r is not None]
    return {
        "cs_c": sum(conflicted) / n,
        "cs_r": (sum(defined) / len(defined)) if defined else None,
        "n_claims": n,
        "n_conflicted_claims": int(sum(conflicted)),
        "n_claims_without_relevant_evidence": n - len(defined),
    }
