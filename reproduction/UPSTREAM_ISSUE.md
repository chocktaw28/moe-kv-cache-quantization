**Title:** fa3 prefill crashes silently on sm89 with INT2 KV cache — and sm89
support status is internally inconsistent

## Summary

On an RTX 4090 (sm89), running with `--prefill-attention-backend fa3`
**combined with `--kv-cache-dtype int2`**, the server process dies silently —
no traceback, no CUDA error, no non-zero exit. Switching prefill to `triton`
(matching the decode backend the script already defaults to) resolves it, with
everything else held constant.

Separately, while investigating this, I found that the codebase gives three
different, mutually inconsistent answers about whether sm89 is supported by
the vendored FlashAttention-3 build at all — including one that directly
contradicts the crash I observed. I think both are worth a look, and I've
tried to keep them clearly separated below: one is an observed failure, the
other is an internal inconsistency I noticed along the way.

## Environment

- GPU: NVIDIA RTX 4090, 24 GB, compute capability sm89 (Ada Lovelace)
- OSCAR commit: `41ebcdba3db5f0ce1339c3727caea80df575d437`
- torch `2.9.1+cu129`
- Model: Qwen3-8B, `TP_SIZE=1`
- `--kv-cache-dtype int2`, locally calibrated rotations (not RotationZoo)

## Observed failure

With `--prefill-attention-backend fa3` and `--kv-cache-dtype int2`, the server
process exits with no error output of any kind — no Python traceback, no CUDA
error, no core dump, no non-zero exit code visible in the wrapper script.
Switching only `--prefill-attention-backend` to `triton` (decode backend stays
`triton`, everything else unchanged) makes the server start and run
correctly.

**What I have not tested:** fa3 prefill with a **BF16** KV cache (no INT2). So
I can't say whether this is a general fa3/sm89 problem or specific to the
combination with OSCAR's INT2 path — I'm reporting the combination I actually
observed, not a broader claim.

**I no longer have the instance this happened on**, so I can't currently
provide a minimal, re-runnable repro script. If it would help, I'm glad to
reconstruct one on a fresh pod — let me know and I'll prioritize it.

## The three-way inconsistency on sm89 support

Independent of the crash above, I went looking in
`sglang-research/python/sglang/jit_kernel/flash_attention_v3.py` for whether
sm89 is a supported target for the vendored fa3 build, and found three
statements that don't agree with each other:

**1. A code comment says sm89 is explicitly supported**, naming the 4090
directly (`flash_attention_v3.py:79`, inside `_is_fa3_supported()`; duplicated
verbatim at `jit_kernel/tests/test_flash_attention_3.py:31`):

> ```
> #  And for sgl-kernel right now, we can build fa3 on sm80/sm86/sm89/sm90a.
> #  That means if you use A100/A*0/L20/L40/L40s/4090 you can use fa3.
> ```

**2. The runtime error string says sm89 is excluded**, requiring sm90+
(`flash_attention_v3.py:125` and `:196`, identical text at both call sites):

> ```python
> raise NotImplementedError(
>     "flash_attn at sgl-kernel is only supported on sm90 and above"
> )
> ```

**3. The test-skip reason gives a third, different boundary**
(`jit_kernel/tests/test_flash_attention_3.py:471` and `:1043`, identical text
at both call sites):

> ```python
> reason="flash_attn at sgl-kernel is only supported on sm90 or sm80",
> ```

Three different statements about the same boundary, in the same file (plus its
test), none of them agreeing.

**What the code actually checks — and it agrees with the comment, not the
error strings:**

```python
def _is_fa3_supported(device=None) -> bool:
    ...
    if torch.version.cuda is None:
        return False
    return (torch.version.cuda >= "12.3") and (
        torch.cuda.get_device_capability(device)[0] == 9
        or torch.cuda.get_device_capability(device)[0] == 8
    )
```

`get_device_capability()[0]` returns only the **major** compute-capability
number. sm89 has major version 8, so `== 8` is `True` — this function returns
`True` on an RTX 4090. That matches statement 1 and contradicts statements 2
and 3: the `NotImplementedError` and the skip reason both describe a
"sm90-and-above" boundary that the actual guard does not enforce. On sm89,
those two error strings are unreachable — the condition that would raise them
never fires.

## Note: the capability guard passes on sm89

Given the above, `_is_fa3_supported()` returns `True` on sm89, so it did not
reject my run. If it *had* fired on the prefill path in my configuration, I
would have gotten a clear `NotImplementedError` instead of a silent process
death. So while I can't state that the crash I observed *is* an sm90-guard
issue — the guard's own logic says sm89 passes — the fact that a
capability-based guard exists at all, and simply wasn't the thing that caught
this particular failure, seems like the most actionable observation here. A
silent death downstream of a passing capability check is worse than either a
clean pass or a clean rejection.

## Possible mechanism (speculative)

**I have not verified this and offer it only as a starting point** — I did not
profile the kernel, did not capture shared-memory usage, and did not test fa3
with a BF16 KV cache. This is a hypothesis, not a diagnosis.

The same comment block quoted above (`flash_attention_v3.py:74-78`) contains
two more lines I haven't cited yet, and they're the only lead I have toward a
mechanism:

> ```
> #  FA3 can fail without a enough shared memory for a some shapes, such as higher
> #  hidden_dim or some special cases.
> #  Right now, fa3 is supported for sm80/sm87 and sm86/sm89. The main different
> #  Between sm80/sm87 and sm86/sm89 is the shared memory size.
> ```

Two things in there, put together:

- The authors' own comment documents that fa3 can fail on insufficient shared
  memory for certain shapes.
- The same comment identifies shared memory size as *the* main difference
  between sm80/sm87 and sm86/sm89 — and this hardware (RTX 4090) is sm89.

Separately, OSCAR's INT2 KV-cache path changes the memory layout the attention
kernel operates on relative to a standard BF16 cache.

Put together, a shared-memory limit specific to sm89's smaller budget, tripped
by the INT2 path's layout for some shape in this configuration, is one
candidate explanation for a failure that produces no error output. I want to be explicit that this is speculation built from a code
comment, not something I measured. If it's useful, I'd guess `compute-sanitizer
--tool memcheck` or `cuda-gdb` on a repro would settle it quickly for someone
with the hardware and time; I don't currently have either.

## Workaround

Force both prefill and decode to `triton`:

```diff
 SERVER_ARGS=(
     --model-path "${MODEL}"
     --tensor-parallel-size "${TP_SIZE}"
-    --prefill-attention-backend fa3
+    --prefill-attention-backend triton
     --decode-attention-backend triton
     --kv-cache-dtype int2
     --kv-cache-quant-group-size "${GROUP_SIZE}"
     ...
```

Cost: first server start with a new config takes ~320s for Triton kernel
compilation (cached on subsequent starts with the same config).

## Suggestions

1. **Reconcile the three support statements.** At minimum, the comment at
   `flash_attention_v3.py:79` and the error text at `:125`/`:196` should agree
   with each other and with what `_is_fa3_supported()` actually checks. Right
   now a reader has no way to know which of the three to trust without reading
   the boolean expression themselves.
2. **Consider extending the capability/config guard to cover the prefill +
   INT2-KV-cache combination**, if that combination is in fact unsupported on
   sm89 — so it fails loudly with the existing `NotImplementedError` pathway
   instead of silently. I can't point to the exact code path responsible for
   the crash since I don't have a repro to trace it with, so I'd defer to
   whoever knows the prefill/INT2 interaction best on where such a check
   would belong.

## Minor: dead configuration variable

`rotation/eval_oscar_gpqa.sh` documents, in its own header comment (line 21):

```
#   PRE_ROPE_FA3   set to 1 to force prefill fa3 + decode triton (default 1)
```

I checked with a repo-wide `grep -rn PRE_ROPE_FA3` and this comment is the
**only** match in the tree — the variable is not read anywhere. The backends
are hardcoded directly in `SERVER_ARGS` a few lines below. Setting
`PRE_ROPE_FA3` currently has no effect, and the comment suggests a
configuration path that doesn't exist. Worth either wiring it up or removing
the comment.

## Offer

I'm reproducing the Qwen3-8B baseline on consumer hardware and building on
this project's work, not filing a complaint — happy to open a PR for either
the documentation reconciliation or the dead-variable cleanup if that's
useful, and to help reconstruct a minimal repro on a fresh pod if that would
make the first issue actionable.
