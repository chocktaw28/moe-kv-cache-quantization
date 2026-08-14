# Run Summary — MoE / Edge LLM Quantization Literature Corpus

Run date: 2026-08-14
Sources: arXiv API (export.arxiv.org), Semantic Scholar Graph API (`/paper/search/bulk`)
Date range applied: 2024-01-01 to 2026-08-14

## 1. Papers found per cluster (pre-dedup)

Counts are unique papers per source per cluster, after each source's own relevance filtering (see Section 5 for the Semantic Scholar relevance-filter issue), *before* cross-source dedup.

| Cluster | arXiv unique papers | Semantic Scholar unique papers |
|---|---|---|
| A — MoE quantization (core) | 13 | 54 |
| B — On-device/edge LLM quantization | 44 | 47 |
| C — Named baseline methods | 92 | 218 |
| D — KV cache / sub-4-bit / extreme quant | 122 | 134 |

- Total unique arXiv papers (any cluster, in date range): **260**
- Total unique Semantic Scholar papers (any cluster, in date range, post relevance-filter): **431**
- A paper can appear in more than one cluster, so these do not sum to source totals.

## 2. Deduplication

- Combined source records entering dedup: 691 (260 arXiv + 431 SS)
- **Unique papers after dedup: 566**
- Merge events: 125 (breakdown: exact arXiv-ID match, DOI exact match, arXiv-ID derived from a `10.48550/arXiv.{id}`-pattern DOI backfilled on Semantic Scholar records that had no populated `arxiv_id` field, and fuzzy title match ≥0.90 similarity)
- Near-miss fuzzy matches (0.80–0.90 similarity) logged but NOT merged: 7 — see `logs/dedup_log.jsonl` for the specific title pairs and ratios, for manual audit.
- **Process note (self-correction):** the first dedup pass produced 572 unique papers and missed several same-paper duplicates where a Semantic Scholar record had `arxiv_id: null` but a DOI encoding the arXiv ID (e.g. `10.48550/arXiv.2406.08155`) that didn't cross the fuzzy-title threshold against its arXiv-sourced sibling (e.g. "QuantMoE-Bench: Examining..." vs "Examining...: A Benchmark" scored 0.814, below the 0.90 threshold). This was caught during extraction QA (a sub-agent flagged the same paper appearing under two different `paper_id`s) and fixed by adding an arXiv-DOI-pattern recognition rule to the dedup script, then re-running dedup end-to-end. The corrected run produced 566 unique papers. All dedup log entries in `logs/dedup_log.jsonl` reflect the corrected, final run.

## 3. Triage (coordinator-directed, applied after dedup)

Per explicit coordinator decision, this rule was applied to the 566 deduped papers before Step 3 extraction:

- **Clusters A and D: NO CAP.** Any paper whose cluster_hits include A ("MoE quantization" core target) or D (KV cache, sub-4-bit, extreme quantization) was kept unconditionally and got a full-text extraction attempt.
- **Clusters B and C: CAPPED at ~15 papers per 6-month bucket**, where bucket = submission/publication date bucketed into {2024H1, 2024H2, 2025H1, 2025H2, 2026H1, 2026H2(partial, through 2026-08-14)}, and selection within each bucket was by **citation count descending** (ties broken by title for determinism). This cap applied **only** to papers whose cluster_hits were confined to {B, C} with no A or D hit — any B/C paper that also matched A or D was treated as an uncapped A/D paper.
- Capped B/C-only papers that survived the cap got **abstract+tldr-only extraction** (no full-text attempt), with `extraction_confidence` defaulted to "low" unless the abstract was unusually explicit.
- Excluded B/C-only papers (below the cap in their bucket) were **not** extracted into `papers.csv`, but every exclusion was logged to `logs/triage_log.jsonl` with paper_id, title, cluster(s), citation_count, bucket, and rank-of-total.

**Bucket-by-bucket B/C-only breakdown (post-dedup-fix):**

| Bucket | Total B/C-only candidates | Kept (≤15 cap) | Excluded |
|---|---|---|---|
| 2024H1 | 30 | 15 | 15 |
| 2024H2 | 21 | 15 | 6 |
| 2025H1 | 55 | 15 | 40 |
| 2025H2 | 49 | 15 | 34 |
| 2026H1 | 117 | 15 | 102 |
| 2026H2 (partial) | 19 | 15 | 4 |
| **Total** | **291** | **90** | **201** |

- A/D papers (uncapped): **275**
- B/C-only papers kept (within cap): **90**
- **Total final corpus: 365 papers**
- B/C-only papers excluded by the cap: **201** (logged individually in `logs/triage_log.jsonl`, reason string: `"excluded: below citation-count cap for cluster B/C in bucket X, rank Y of Z"`)

## 4. Unique papers after dedup and triage

**Final corpus size: 365 papers** (rows in `data/papers.csv`), all with unique `paper_id`.

## 5. Extraction confidence

| Confidence | Count |
|---|---|
| high | 30 |
| medium | 183 |
| low | 152 |

`high` was reserved for papers where real ar5iv full-text was fetched and read, with clear explicit statements across most fields. `medium` covers both (a) full-text reads with some ambiguity/gaps, and (b) unusually explicit/detailed abstract-only extractions. `low` is the default for abstract-only extraction and any case involving genuine ambiguity — per the task's strict rule, `extraction_confidence` was never set to `high` for abstract-only records, and `not stated` was used literally wherever a field wasn't explicitly supported by source text rather than guessed.

## 6. Full-text vs abstract-only extraction

- **Priority set attempted for fuller-text extraction: 64 papers** (61 Cluster-A hits + 3 Cluster-D-only papers whose abstract clearly discussed MoE/expert architectures).
- Of those 64, **~47 achieved real full-text extraction via ar5iv HTML mirrors** (methods/experiments/limitations sections read directly, not just the abstract).
- The remaining ~17 priority papers fell back to abstract-only extraction because: no arXiv ID (DOI-only, e.g. IEEE/ACM/Springer venues with no open full text), ar5iv mirror not yet available for very recent submissions (redirect/404), or the source page was JS-rendered/paywalled and could not be resolved with confidence (in one case a plausible OpenReview match was found but deliberately NOT attributed, to avoid fabrication — logged as low-confidence, needs manual review).
- **All other 301 papers (peripheral Cluster B/C-only survivors, plus any Cluster-D-only papers not flagged as MoE-relevant) used abstract+metadata-only extraction**, per policy, with no WebFetch/full-text attempt made (deliberate, to keep the corpus tractable at this scale).
- Total: **~50 full-text extractions, ~315 abstract-only extractions** (approximate split; a handful of borderline "used arXiv abs page for comment/venue metadata but not full body text" cases resist a clean binary count — see individual `extraction_notes` per row).

## 7. API errors / failures

- **Semantic Scholar `/paper/search/bulk` — hard failure on first attempt, all 20 queries, HTTP 400 Bad Request.** Root cause: the `fields` parameter included `tldr`, which is not a supported field on the bulk endpoint (confirmed via direct curl reproduction: `{"error":"Unrecognized or unsupported fields: [tldr]"}`). This means the task's originally-intended `tldr`-assisted extraction for peripheral papers was not available — abstract text was used as the primary extraction source instead throughout. Fixed by removing `tldr` from the fields list and re-running all 20 queries, which then succeeded (0 errors on rerun). The original 20 failed-query log entries remain in `logs/query_log.jsonl` for audit purposes rather than being deleted.
- **Semantic Scholar `/bulk` endpoint does not relevance-rank results** (confirmed empirically): broad terms like "2-bit LLM quantization" returned 432 hits and "mixture of experts quantization" returned 161, many clearly off-topic (e.g. wireless channel modeling, audio codecs, PII substitution papers) that merely shared incidental keyword overlap. A title-level relevance filter (quantization-signal term + cluster-appropriate topical-signal term, both required to appear in the *title*, not just the abstract body) was built and applied; papers with signal only in the abstract body (not title) were excluded from promotion to per-paper raw records but their raw batch JSON was preserved verbatim, so nothing is unrecoverable. This filtering logic and its exclusions are logged in `logs/ss_relevance_log.jsonl`.
- **arXiv strict-phrase queries returned 0 results for several terms** on the first pass ("sparse MoE low-bit", "QuaRot quantization", "GGUF GGML quantization", "2-bit LLM quantization", "sub-4-bit LLM quantization", "extreme quantization large language models", plus two edge/mobile B-cluster terms) — arXiv's `all:"exact phrase"` matching combined with category filters is strict. These were re-run as broadened AND-of-keywords queries (e.g. `all:"sparse mixture of experts" AND all:quantization`), all logged separately in `logs/query_log.jsonl` with `(broadened: ...)` in the query_term field, and all returned useful hits on retry.
- No arXiv rate-limit (429) errors encountered; no Semantic Scholar 429s encountered on the corrected run (only the field-name 400s described above).
- 2 priority papers had genuinely unreachable full text (IEEE Xplore DOIs with no accessible abstract page content) and were left at low-confidence abstract-only.

## 8. Papers flagged as ambiguous or borderline-relevant

A non-exhaustive curated list of the more substantive judgment calls surfaced during extraction (full detail in each row's `extraction_notes` in `data/papers.csv`):

- **2411.19402** ("On the Role of Discrete Representation in Sparse Mixture of Experts") — "quantization" here means vector-quantized discrete routing codebooks (VQ-VAE-style), not numeric bit-width weight quantization. Kept in corpus (matched Cluster A search) but clearly a different sense of "quantization" than the rest of the corpus; flagged prominently.
- **2503.21135** — batch metadata title/abstract call the method "DynaMo," but the actual current arXiv full text (v3) names the method "MoQa" and never mentions "DynaMo" — apparent paper-retitling between versions not reflected in the cached abstract metadata. Confidence downgraded to low; flagged for manual verification.
- **2601.13563** ("ButterflyMoE") — the JSON metadata title read "Sub-Linear Ternary Experts" but the live arXiv page shows "Compression-Scalable Ternary Experts" — likely a title revision between versions; flagged, not silently corrected without a note.
- **2601.22001**, **2606.25092**, **2607.06601**, **2602.11937** — all matched the core Cluster A search but use quantization only as a secondary/example mechanism within a broader systems, causal-analysis, or routing-search contribution rather than as their central research question. Kept (they did surface via the intended search terms) but flagged as borderline-central.
- **2605.14359** (RQ-MoE) and **2602.24059** (Quant Experts) — "Mixture of Experts" in these titles refers to an internal architectural component (embedding-quantization routing; LoRA-adapter compensation) rather than the paper testing an actual MoE *language model* — marked `is_moe: false` despite the title.
- **2607.14334** (MixCompress) — Mixture-of-Experts applied to learned *image* compression, not LLM quantization at all; likely a keyword-overlap false positive from the search.
- **2511.11743** ("Curiosity-Driven QMoE") — confirmed via full text to be about small-scale audio classification models (1–5M params), not LLMs.
- **2408.15305** (sLAVA) and **2508.15706** (SparseLoCo) — both use "mixture of experts" loosely (PEFT/LoRA-style adapter mixtures; MoE as one of several tested pretraining scales) rather than testing genuine large sparse-MoE LLMs; marked `is_moe: false`.
- **2507.07145** (CCQ) — this Cluster-D-only (KV-cache-search-derived) paper actually tests genuine large MoE models (DeepSeek-V3 671B, ERNIE-4.5-300B-A47B) compressed to 2–2.75 bits. It survived triage as a B/C/D-peripheral paper and got abstract-only extraction; a human reviewer may want to manually promote this one for a fuller-text read given its clear substantive relevance to the core MoE-quantization question despite not having matched Cluster A in search.
- A systemic pattern, not a single-paper issue: the large majority of Cluster D ("KV cache quantization" / "2-bit" / "sub-4-bit") search hits target **dense** LLMs (LLaMA/Qwen/Mistral/Gemma families), confirming that KV-cache quantization and extreme low-bit weight quantization research is currently concentrated on dense architectures — MoE-specific treatment of these adjacent mechanisms is comparatively sparse in this corpus, which is itself a substantive (if expected) observation for the scoping question, even though this run was not asked to editorialize on it further.

## 9. Notes on process integrity

- Extraction was performed by parallel sub-agents working from a shared, pre-fetched raw-JSON corpus (never re-querying the live APIs during extraction), each given the same strict no-fabrication rules and required to write "not stated" rather than infer. Several agents independently caught and refused to report apparent fabrications surfaced by intermediate tooling (e.g., an ar5iv-derived summary hallucinating a venue name or a GitHub URL) — those refusals are preserved in the relevant rows' `extraction_notes`.
- All raw API responses (per-query batch files and per-paper JSON records) are retained verbatim in `raw/`, including the batch files from the failed initial Semantic Scholar attempt's query log entry (though the failed attempt itself produced no batch files, since the request errored before any body was returned) — nothing needs to be re-fetched to re-derive or audit any extraction in `data/papers.csv`.
