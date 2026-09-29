import importlib
import os
import sys
from dataclasses import asdict
from unittest.mock import patch
from contextlib import ExitStack

import pytest
import torch

from matformer import GPT, GPTConfig, generate


@pytest.fixture
def model():
    torch.manual_seed(7)
    model = GPT(GPTConfig(sequence_len=16, vocab_size=67, n_layer=4,
                          n_embd=32, n_head=4, n_kv_head=4, head_dim=8,
                          window_pattern="L", matformer_ffn_hidden_dims=(8, 16, 32, 64),
                          matformer_head_counts=(4, 4, 4, 4)))
    with torch.no_grad():
        model.smear_lambda.fill_(0.4)
        model.x0_lambdas.fill_(0.15)
    return model.eval()


@pytest.mark.parametrize("width", ["S", "M", "L", "XL"])
@pytest.mark.parametrize("chunks", [(1, 1, 1, 1, 1, 1), (3, 1, 1, 1), (2, 2, 2)])
def test_cache_matches_full_prefix(model, width, chunks):
    ids = torch.tensor([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]])
    expected = model(ids, ffn_hidden_dim=width)
    cache = model.new_cache(batch_size=2)
    outputs, pos = [], 0
    for size in chunks:
        outputs.append(model(ids[:, pos:pos + size], kv_cache=cache, ffn_hidden_dim=width))
        pos += size
    torch.testing.assert_close(torch.cat(outputs, dim=1), expected, rtol=1e-5, atol=1e-6)
    assert cache.get_pos() == ids.size(1)


def test_switching_matches_uncached_per_token_schedule(model):
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    widths = ["S", "S", "XL", "M", "L", "S"]
    cache = model.new_cache()
    parts = [model(ids[:, :2], kv_cache=cache, ffn_hidden_dim="S")]
    old_k = cache.k_cache[:, :, :2].clone()
    old_v = cache.v_cache[:, :, :2].clone()
    pointer = cache.k_cache.data_ptr()
    for i in range(2, ids.size(1)):
        parts.append(model(ids[:, i:i+1], kv_cache=cache, ffn_hidden_dim=widths[i]))
    assert cache.k_cache.data_ptr() == pointer
    torch.testing.assert_close(cache.k_cache[:, :, :2], old_k, rtol=0, atol=0)
    torch.testing.assert_close(cache.v_cache[:, :, :2], old_v, rtol=0, atol=0)
    # Independent full-sequence oracle: apply the scheduled FFN to each position
    # in every layer, with ordinary causal attention and no cache at all.
    with ExitStack() as stack:
        for block in model.transformer.h:
            original = block.mlp.forward

            def scheduled(x, ffn_hidden_dim=None, original=original):
                return torch.cat([original(x[:, i:i+1], model.widths[w])
                                  for i, w in enumerate(widths)], dim=1)

            stack.enter_context(patch.object(block.mlp, "forward", scheduled))
        expected = model(ids)
    actual = torch.cat(parts, dim=1)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert not torch.allclose(actual[:, -1], model(ids, ffn_hidden_dim="S")[:, -1])


def test_cache_limits_reset_and_bad_width(model):
    ids = torch.tensor([[1, 2]])
    cache = model.new_cache(max_seq_len=2)
    model(ids, kv_cache=cache, ffn_hidden_dim="M")
    with pytest.raises(ValueError, match="capacity"):
        model(ids[:, :1], kv_cache=cache)
    cache.reset()
    assert cache.prev_embedding is None and cache.get_pos() == 0
    torch.testing.assert_close(model(ids, kv_cache=cache), model(ids))
    with pytest.raises(ValueError, match="Width must"):
        model(ids, ffn_hidden_dim="unknown")
    with pytest.raises(ValueError, match="outside"):
        model(ids, ffn_hidden_dim=65)


def test_generate_policy_and_one_token_prompt(model):
    calls = []

    def policy(step, ids):
        calls.append((step, len(ids)))
        return "S" if step < 2 else "XL"

    result = list(generate(model, [1], max_new_tokens=4, width=policy))
    assert len(result) == 4
    assert calls == [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert list(generate(model, [1], max_new_tokens=0)) == []
    sampled = list(generate(model, [1], max_new_tokens=4, temperature=0.8, seed=7))
    assert sampled == list(generate(model, [1], max_new_tokens=4, temperature=0.8, seed=7))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_original_model_parity(model, dtype):
    source = os.environ.get("NANOCHAT_SOURCE_DIR")
    if not source:
        pytest.skip("Set NANOCHAT_SOURCE_DIR to the original repo for source parity")
    sys.path.insert(0, source)
    try:
        reference = importlib.import_module("nanochat.gpt_matformer")
    finally:
        sys.path.pop(0)
    reference.COMPUTE_DTYPE = dtype
    for parameter in model.parameters():
        if parameter.ndim >= 2:
            parameter.data = parameter.data.to(dtype)
    model.cos, model.sin = model._precompute_rotary_embeddings(model.rotary_seq_len, model.head_dim)
    original = reference.GPT(reference.GPTConfig(**asdict(model.config))).eval()
    original.load_state_dict(model.state_dict(), strict=True, assign=True)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    with torch.inference_mode():
        for width in model.widths.values():
            torch.testing.assert_close(model(ids, ffn_hidden_dim=width),
                                       original(ids, ffn_hidden_dim=width), rtol=0, atol=0)
        a, b = model.new_cache(), model.new_cache()
        for begin, end, width in [(0, 3, 8), (3, 4, 64), (4, 5, 16), (5, 6, 32)]:
            torch.testing.assert_close(
                model(ids[:, begin:end], kv_cache=a, ffn_hidden_dim=width),
                original(ids[:, begin:end], kv_cache=b, ffn_hidden_dim=width), rtol=0, atol=0)


@pytest.mark.skipif(os.environ.get("RUN_HF_TESTS") != "1", reason="Downloads the ~3 GB checkpoint")
def test_published_checkpoint():
    from matformer import load_checkpoint

    model, tokenizer, metadata = load_checkpoint(device="cpu")
    assert metadata["step"] == 3624
    assert model.widths == {"S": 640, "M": 1280, "L": 2560, "XL": 5120}
    ids = torch.tensor([[tokenizer.encode_single_token("<|bos|>")]
                        + tokenizer.encode_ordinary("The future of AI is")])
    for width in model.widths:
        expected = model(ids, ffn_hidden_dim=width)
        cache = model.new_cache(max_seq_len=ids.size(1))
        actual = torch.cat([model(ids[:, :-1], kv_cache=cache, ffn_hidden_dim=width),
                            model(ids[:, -1:], kv_cache=cache, ffn_hidden_dim=width)], dim=1)
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
    outputs = list(generate(model, ids[0].tolist(), max_new_tokens=4,
                            width=lambda step, _: "S" if step < 2 else "XL"))
    assert len(outputs) == 4
    assert tokenizer.decode(outputs)
