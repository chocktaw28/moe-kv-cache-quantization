# smoke_prompt.txt — provenance

**sha256:** `ec196d243a74c64150d6738272198fa5149250c23a54708213ba34aa2877b9c5`
**size:** 26,819 characters / 4,411 whitespace words

## Composition

Two parts, concatenated in fixed order with no randomness:

1. **Public-domain literary text.** The opening passages of *Moby-Dick; or, The
   Whale* by Herman Melville (1851). Public domain worldwide; the work predates
   any surviving copyright term and is distributed by Project Gutenberg as
   ebook #2701. Reproduced here as running prose to give the model natural,
   coherent text to continue.

2. **Deterministic labelled filler.** 60 numbered "Observation N." sentences
   cycling through 6 fixed maritime topics, appended to reach the length
   target. These are neutral generated filler, clearly labelled as such, and
   are not quoted from any source.

## Why this text

The smoke test measures kernel liveness, not knowledge. It needs a prompt that
is long, deterministic, and produces *free-form prose* whose degradation is
visible to a human reader. Literary narrative is well suited: a broken INT2
path shows up as repetition, incoherence, or token salad, which is obvious at a
glance in a way that a multiple-choice letter never is.

## Length

Token count is **asserted at runtime by the smoke script using the model's own
tokenizer** (`AutoTokenizer.from_pretrained(MODEL)`), not assumed here.

Estimated 5,700–6,700 tokens by two independent heuristics (chars/4 and
words×1.3). This was **not** verified against the Qwen3 tokenizer when the file
was written — `transformers` was not installed on the machine that generated it
— so the figures above are estimates. The runtime assertion in
`_common.sh::assert_smoke_prompt_length` is the authoritative check and aborts
if the real count does not exceed sink+recent.

The margin is deliberately large. Even if the true count were half the low
estimate, it would still exceed the paper-knee window (64 + 256 = 320) by
roughly 9×, so the INT2 path is genuinely exercised rather than sitting
entirely inside the high-precision windows.

## Regenerating

The file is committed and should not need regenerating. If it is ever rebuilt,
the builder must remain deterministic — all three arms depend on byte-identical
input, and a changed prompt invalidates cross-arm comparison.
