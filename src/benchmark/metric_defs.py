"""Native metric definitions, copied from each implementation's code/README (not re-derived)."""

# key: (evaluator, metric) -> (definition, higher_is_better, range, reference_dependent, evidence_based)
DEFS = {
    ("ConflictScore", "cs_c"): (
        "PUBLISHED CS-C (paper Sec.3): fraction of the response's claims with >=1 supporting and >=1 contradicting doc",
        False, "[0,1]", False, True),
    ("ConflictScore", "cs_r"): (
        "PUBLISHED CS-R (paper Sec.3): mean over claims of |D-|/(|D+|+|D-|); claims with no relevant doc excluded (undefined in paper)",
        False, "[0,1]", False, True),
    ("ConflictScore", "n_conflicted_claims"): ("number of claims with both supporting and contradicting docs", None, "count", False, True),
    ("ConflictScore", "n_claims_without_relevant_evidence"): ("claims with no SUPPORTS/CONTRADICTS doc (excluded from CS-R)", None, "count", False, True),
    ("ConflictScore", "n_claims"): ("number of extracted claims", None, "count", False, False),
    ("RAGChecker", "precision"): ("share of response claims entailed by gt_answer", True, "[0,1]", True, False),
    ("RAGChecker", "recall"): ("share of gt_answer claims entailed by response", True, "[0,1]", True, False),
    ("RAGChecker", "f1"): ("harmonic mean of precision and recall (0 if either is 0)", True, "[0,1]", True, False),
    ("RAGChecker", "claim_recall"): ("share of gt claims entailed by >=1 retrieved chunk", True, "[0,1]", True, True),
    ("RAGChecker", "context_precision"): ("share of chunks entailing >=1 gt claim", True, "[0,1]", True, True),
    ("RAGChecker", "context_utilization"): ("of gt claims present in context, share also entailed by response",
                                            True, "[0,1]", True, True),
    ("RAGChecker", "noise_sensitivity_in_relevant"): (
        "share of response claims that are incorrect (not entailed by gt) yet entailed by a relevant chunk",
        False, "[0,1]", True, True),
    ("RAGChecker", "noise_sensitivity_in_irrelevant"): (
        "share of response claims incorrect yet entailed only by irrelevant chunks", False, "[0,1]", True, True),
    ("RAGChecker", "hallucination"): ("share of response claims incorrect and not entailed by any chunk",
                                      False, "[0,1]", True, True),
    ("RAGChecker", "self_knowledge"): ("share of response claims correct but not entailed by any chunk",
                                       None, "[0,1]", True, True),
    ("RAGChecker", "faithfulness"): ("share of response claims entailed by >=1 retrieved chunk", True, "[0,1]", False, True),
    ("Q-CARE", "completeness"): ("sub-queries covered by the answer's claims / sub-queries", True, "[0,1]", False, False),
    ("Q-CARE", "conciseness"): ("claims that serve a covered sub-query / claims", True, "[0,1]", False, False),
    ("Q-CARE", "verifiableness"): ("claims supported by >=1 retrieved chunk / claims", True, "[0,1]", False, True),
    ("Q-CARE", "precision_at_10"): ("C-Prec@10: mean graded chunk relevance over top-10 (max = n_chunks/10)",
                                    True, "[0,1]", False, True),
    ("Q-CARE", "ndcg_at_10"): ("C-nDCG@10 over graded relevance, linear gain", True, "[0,1]", False, True),
    ("Q-CARE", "num_atomic_facts"): ("number of atomic facts extracted from the answer", None, "count", False, False),
}
