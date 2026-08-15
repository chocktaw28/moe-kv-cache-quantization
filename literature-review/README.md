# Literature Review

A structured review of KV-cache quantization research for Mixture-of-Experts (MoE)
LLMs, compared against the much larger body of work on dense-model KV-cache
quantization.

Phase 0 of the [MoE KV-Cache Quantization](../) project.

## Corpus

365 papers (2024–2026), built via arXiv and Semantic Scholar API queries with
structured field extraction under a strict no-fabrication policy — every field not
explicitly stated in a source paper is marked "not stated" rather than inferred.

See [`run_summary.md`](run_summary.md) for full methodology, including
deduplication, triage rules, and known search-coverage limitations.

## Key finding

Only **6 of 67** confirmed-MoE papers in the corpus address KV-cache quantization
at all, versus **117** dense-model papers on the same topic.

This asymmetry was checked against two alternative explanations before being
treated as a real gap:

1. **Search-design artifact** — the query strategy systematically missing MoE
   KV-cache work rather than the work not existing.
2. **Corpus-labelling error** — papers misclassified on the `is_moe` dimension.

It holds up under manual verification of both. See
[`coverage_matrix_primary.md`](coverage_matrix_primary.md) and
[`coverage_matrix_reference.md`](coverage_matrix_reference.md) for the underlying
data.

Ambiguous architecture classifications were resolved by hand using outside
knowledge of model architectures — the extraction agent was restricted to
in-document information only, so papers testing models whose MoE status isn't
stated in the text required manual resolution. That process is logged in
[`is_moe_reclassification.jsonl`](is_moe_reclassification.jsonl).

## Contents

| File | Description |
|---|---|
| [`run_summary.md`](run_summary.md) | Corpus construction methodology |
| [`coverage_matrix_primary.md`](coverage_matrix_primary.md) | Bit-width × architecture × hardware breakdown, confirmed-MoE papers only |
| [`coverage_matrix_reference.md`](coverage_matrix_reference.md) | Same breakdown for dense/reference papers, for comparison |
| [`recency_analysis.md`](recency_analysis.md) | Publication-rate and author-overlap analysis |
| [`is_moe_reclassification.jsonl`](is_moe_reclassification.jsonl) | Manual verification log for ambiguous architecture classifications |

## Scope

This directory contains the corpus methodology and coverage analysis only. The gap
analysis derived from it, and the research direction selected as a result, are held
privately while the work is in progress.
