# MoE KV-Cache Quantization — Literature Review

A structured literature review examining KV-cache quantization research for
Mixture-of-Experts (MoE) LLMs, compared against the much larger body of work
on dense-model KV-cache quantization.

## Corpus

365 papers (2024–2026), built via arXiv and Semantic Scholar API queries with
structured field extraction under a strict no-fabrication policy — every field
not explicitly stated in a source paper is marked "not stated" rather than
inferred. See `literature-review/run_summary.md` for full methodology,
including deduplication, triage rules, and known search-coverage limitations.

## Key finding

Only 6 of 67 confirmed-MoE papers in the corpus address KV-cache quantization
at all, versus 117 dense-model papers on the same topic. This asymmetry was
checked against two alternative explanations (search-design artifact,
corpus-labelling error) and holds up under manual verification — see
`literature-review/coverage_matrix_primary.md` and
`literature-review/coverage_matrix_reference.md` for the full data.

## Repository structure

- `literature-review/run_summary.md` — corpus construction methodology
- `literature-review/coverage_matrix_primary.md` — bit-width × architecture ×
  hardware breakdown, confirmed-MoE papers only
- `literature-review/coverage_matrix_reference.md` — same breakdown, dense/
  reference papers, for comparison
- `literature-review/recency_analysis.md` — publication-rate and
  author-overlap analysis
- `literature-review/is_moe_reclassification.jsonl` — manual verification log
  for ambiguous architecture classifications
