# nanochat MatFormer inference

Minimal PyTorch inference for [atrost/nanochat-d24-matformer-ffnlog-fullattn-20260507](https://huggingface.co/atrost/nanochat-d24-matformer-ffnlog-fullattn-20260507): load the parent checkpoint, select S/M/L/XL, and change FFN width during autoregressive decoding while retaining the KV cache.

The implementation is extracted from `SprocketLab/nanochat-matryoshka`, preserving checkpoint parameter names and model operations. This repository contains inference, examples, and tests only. It has no dependency on the original repository, training code, datasets, optimizers, or custom CUDA kernels.

## Install and run

Requires Python 3.10+ and PyTorch 2.6+. From this repository:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Fixed width (also accepts M, L, XL).
python -m matformer --width S --prompt "The future of artificial intelligence is"

# Predict 16 tokens at S, then use XL, keeping the same cache.
python -m matformer --width S --switch-to XL --switch-after 16
```

Defaults to CUDA/bfloat16 where supported, otherwise CPU/float32. Use `--device cpu` or explicit `--device cuda --dtype bfloat16`. The first run downloads only `model_003624.pt` (about 3 GB), its metadata, and `tokenizer.pkl`; it does not download optimizer states or other checkpoints. Files are cached by Hugging Face. The Hub revision is pinned to `d96d29fa20dc53abbd0830bd7f8ced8e9c123e77`.

If the Xet transfer backend stalls, prefix the command with `HF_HUB_DISABLE_XET=1` to use HTTP downloads.

This is a **base language model**, so the example uses plain-text completion with a BOS token, not a chat template. Greedy decoding is the default; use `--temperature 0.8 --seed 42` for sampling.

## Widths

The published checkpoint has 24 layers, residual width 1280, context length 2048, and head dimension 80. Its metadata specifies:

| Label | FFN hidden width | Query heads | KV heads |
| --- | ---: | ---: | ---: |
| S | 640 | 16 | 16 |
| M | 1280 | 16 | 16 |
| L | 2560 | 16 | 16 |
| XL | 5120 | 16 | 16 |

Every MLP uses the first `width` rows of its input projection and the first `width` columns of its output projection, with the original squared-ReLU activation. Attention, residual width, token/value embeddings, and output head remain full size. All parent weights remain resident: smaller widths reduce FFN computation, not model storage or KV-cache size. Integer widths from 0 through 5120 are also accepted for experiments; only the four listed widths are documented training granularities.

## Python: fixed width and dynamic policies

```python
from matformer import load_checkpoint, generate

model, tokenizer, metadata = load_checkpoint()
bos = tokenizer.encode_single_token("<|bos|>")
prompt = [bos] + tokenizer.encode_ordinary("The future of AI is")

for width in ("S", "M", "L", "XL"):
    output = list(generate(model, prompt, max_new_tokens=32, width=width,
                           stop_token_id=bos))
    print(width, tokenizer.decode(output))

# A policy can inspect the step and all token IDs seen so far.
def policy(step, token_ids):
    return "S" if step < 16 else "XL"

output = list(generate(model, prompt, max_new_tokens=64, width=policy,
                       stop_token_id=bos))
print(tokenizer.decode(output))
```

Policy step 0 selects the width for prompt prefill and predicts the first output token. Step 1 processes that token and predicts the second. Each forward call uses one width uniformly across layers and across the tokens in that call. Changing the width on the next call takes effect immediately without rebuilding the model or cache.

## Low-level KV-cache reuse

```python
import torch
from matformer import load_checkpoint

model, tokenizer, _ = load_checkpoint()
ids = [tokenizer.encode_single_token("<|bos|>")]
ids += tokenizer.encode_ordinary("The future of AI is")
tokens = torch.tensor([ids], device=model.get_device())
cache = model.new_cache()

# Prefill at S and choose the first generated token.
logits = model(tokens, kv_cache=cache, ffn_hidden_dim="S")
next_token = logits[:, -1].argmax(-1, keepdim=True)

# Switch to XL. Earlier keys/values stay in this same cache.
logits = model(next_token, kv_cache=cache, ffn_hidden_dim="XL")
```

**Yes, this fixed-attention checkpoint supports cache reuse when switching FFN widths.** Each layer's K/V layout is unchanged. The cache also preserves the previous normalized token embedding needed by nanochat's embedding-smear operation. After prefill, pass only new tokens; passing the whole prefix would append it again.

Reusing the cache implements a *mixed-width history*. Earlier tokens keep the keys/values computed under their original widths. Because FFN outputs influence later layers' hidden states, these cached states generally differ from a prefix recomputed entirely at the new width. If an experiment needs that latter baseline, create a fresh cache and prefill the complete token history at the new width. Cache compatibility does not establish quality or speed benefits for a switching policy; those require measurement.

The original model already accepts `ffn_hidden_dim` alongside `kv_cache`. Its generic generation helper does not expose a per-token width policy; the small `generate` helper here adds that interface. This release is scoped to RMSNorm and fixed attention heads, and rejects head-sliced configurations.

## Scope and implementation notes

- PyTorch scaled dot-product attention handles full prefill, single-token decoding, and cached chunks. No FA3 package or kernel download is required.
- The original RoPE, QK normalization/scaling, value-embedding gates, learned residual mixing, embedding smear, backout term, and logit softcap are retained.
- Matrix weights use the requested compute dtype; learned scalars retain their stored precision to preserve the original computation.
- Chunked decoding carries the previous embedding across the chunk boundary; single-token prompts work too. These fix edge cases in the original inference path.
- The cache supports equal-length batches. The generation helper is single-sequence; there is no padding/ragged-batch engine, beam search, tool execution, or context eviction.
- The checkpoint context limit is enforced at 2048 processed tokens. BF16 parent weights alone occupy about 2.12 GB; a full 2048-token, batch-1 BF16 KV cache adds about 252 MB. Runtime memory also includes activations and temporary buffers. CPU float32 uses twice that weight/cache storage.

## Tests

```bash
pip install -e '.[test]'
pytest -q

# Optional: exact small-model parity with a local original checkout.
NANOCHAT_SOURCE_DIR=/path/to/nanochat-matryoshka pytest -q

# Optional: download/load the actual published checkpoint and test all widths.
RUN_HF_TESTS=1 pytest -q
```

Tests cover cached versus uncached logits for all widths, one-token prompts, multi-token chunks, width switching against an uncached per-token-schedule oracle, unchanged cache prefixes, cache reset/bounds, and the generation policy. Large-checkpoint and original-repo tests are opt-in; CI uses only small CPU models.

## Provenance and license

Derived from `nanochat/gpt_matformer.py`, `nanochat/flash_attention.py`, `nanochat/engine.py`, and checkpoint/tokenizer loading in `SprocketLab/nanochat-matryoshka` at commit `e921d5b0313032482ae6590c87ac07757bd389bd`. The last change to the source model file in that checkout is `fe487d064b2690b2db67a6fe16d5dd103c28ca59`. These identify the source used for this extraction, not a verified training-time commit for the Hugging Face checkpoint.

The upstream MIT license and copyright notice are preserved in [LICENSE](LICENSE). Model weights remain hosted on Hugging Face and are not bundled here.
