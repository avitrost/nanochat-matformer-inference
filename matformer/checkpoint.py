"""Load only the final model, metadata, and tokenizer from the published run."""

import json
import pickle

import torch
import tiktoken
from huggingface_hub import hf_hub_download

from .model import GPT, GPTConfig

REPO_ID = "atrost/nanochat-d24-matformer-ffnlog-fullattn-20260507"
REVISION = "d96d29fa20dc53abbd0830bd7f8ced8e9c123e77"
RUN = "d24_matformer_ffnlog_fullattn_20260507"


class _EncodingUnpickler(pickle.Unpickler):
    # The published pickle contains a tiktoken.Encoding, not a nanochat class.
    def find_class(self, module, name):
        if (module, name) == ("tiktoken.core", "Encoding"):
            return tiktoken.Encoding
        raise pickle.UnpicklingError(f"Unexpected tokenizer class: {module}.{name}")


def load_checkpoint(device=None, dtype=None, step=3624, cache_dir=None):
    """Return (model, tiktoken encoding, metadata). Default: CUDA bf16 or CPU fp32."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if dtype is None:
        dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("Expected float32, float16, or bfloat16")

    def download(path):
        return hf_hub_download(REPO_ID, f"{RUN}/{path}", revision=REVISION, cache_dir=cache_dir)

    with open(download(f"checkpoints/meta_{step:06d}.json")) as f:
        metadata = json.load(f)
    config = GPTConfig(**metadata["model_config"])
    with torch.device("meta"):
        model = GPT(config)
    state = torch.load(download(f"checkpoints/model_{step:06d}.pt"),
                       map_location="cpu", weights_only=True, mmap=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True, assign=True)
    # Rotary buffers are nonpersistent: replace the meta buffers before .to().
    model.cos, model.sin = model._precompute_rotary_embeddings(
        model.rotary_seq_len, model.head_dim, device="cpu")
    # Matrix projections already cast to the activation dtype in the source.
    # Keep learned scalars in their stored precision (notably resid_lambdas).
    for parameter in model.parameters():
        parameter.data = parameter.data.to(
            device=device, dtype=dtype if parameter.ndim >= 2 else parameter.dtype)
    model.cos, model.sin = model.cos.to(device), model.sin.to(device)
    # Compute frequencies in fp32, then round directly to the requested dtype.
    model.cos, model.sin = model._precompute_rotary_embeddings(model.rotary_seq_len, model.head_dim)
    model.eval().requires_grad_(False)
    with open(download("tokenizer/tokenizer.pkl"), "rb") as f:
        tokenizer = _EncodingUnpickler(f).load()
    if tokenizer.n_vocab != config.vocab_size:
        raise ValueError("Tokenizer and checkpoint vocabulary sizes differ")
    return model, tokenizer, metadata
