"""Inference subset of nanochat.gpt_matformer; original parameter names preserved."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import attention as flash_attn
from .cache import KVCache

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (quarter context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"
    norm_type: str = "rmsnorm"
    attn_alpha_init_value: float = 1.0
    ffn_alpha_init_value: float = 1.0
    dec_alpha_init_value: float = 1.0
    # Attention head dimension. None preserves the standard n_embd // n_head rule.
    head_dim: int | None = None
    # Uniform MatFormer FFN hidden dimensions for S/M/L/XL-style granularities.
    # The published checkpoint has four widths.
    matformer_ffn_hidden_dims: tuple[int, ...] = ()
    # Uniform MatFormer attention head counts paired with the FFN hidden dims.
    # Empty means "use all heads" for every FFN granularity.
    matformer_head_counts: tuple[int, ...] = ()


def norm(x):
    return F.rms_norm(x, (x.size(-1),)) # note that this will run in bf16, seems ok

class RMSNorm(nn.Module):
    def forward(self, x):
        return norm(x)

class Linear(nn.Linear):
    """Linear projection in the activation dtype."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2

def get_head_dim(config):
    head_dim = config.head_dim if config.head_dim is not None else config.n_embd // config.n_head
    if head_dim <= 0:
        raise ValueError(f"head_dim must be positive, got {head_dim}")
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for rotary embeddings, got {head_dim}")
    if config.n_head * head_dim > config.n_embd:
        raise ValueError(
            f"n_head * head_dim ({config.n_head} * {head_dim}) exceeds n_embd={config.n_embd}"
        )
    return head_dim

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    ve_gate_channels = 12

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = get_head_dim(config)
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.q_width = self.n_head * self.head_dim
        self.kv_width = self.n_kv_head * self.head_dim
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.q_width, self.n_embd, bias=False)
        self.q_norm = RMSNorm()
        self.k_norm = RMSNorm()
        self.qk_scale = 1.2
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, _ = x.size()
        # Project the input to get queries, keys, and values
        # Cache layout: (batch, time, heads, head_dim).
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = self.q_norm(q), self.k_norm(k) # QK norm / point-wise replacement
        q = q * self.qk_scale
        k = k * self.qk_scale

        # PyTorch scaled dot-product attention.
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # Uncached causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        dims = tuple(config.matformer_ffn_hidden_dims) or (4 * config.n_embd,)
        if any(dim < 0 for dim in dims):
            raise ValueError(f"FFN hidden dims must be non-negative, got {dims}")
        if any(a > b for a, b in zip(dims, dims[1:])):
            raise ValueError(f"FFN hidden dims must be sorted ascending, got {dims}")
        self.ffn_hidden_dims = dims
        self.max_ffn_hidden_dim = dims[-1]
        self.c_fc = Linear(config.n_embd, self.max_ffn_hidden_dim, bias=False)
        self.c_proj = Linear(self.max_ffn_hidden_dim, config.n_embd, bias=False)

    def forward(self, x, ffn_hidden_dim=None):
        if ffn_hidden_dim is None:
            ffn_hidden_dim = self.max_ffn_hidden_dim
        assert 0 <= ffn_hidden_dim <= self.max_ffn_hidden_dim, (
            f"ffn_hidden_dim={ffn_hidden_dim} outside [0, {self.max_ffn_hidden_dim}]"
        )
        fc_w = self.c_fc.weight[:ffn_hidden_dim].to(dtype=x.dtype)
        x = F.linear(x, fc_w)
        x = F.relu(x).square()
        proj_w = self.c_proj.weight[:, :ffn_hidden_dim].to(dtype=x.dtype)
        x = F.linear(x, proj_w)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache, ffn_hidden_dim=None):
        x = x + self.attn(self.attn_norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(self.mlp_norm(x), ffn_hidden_dim=ffn_hidden_dim)
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        if config.norm_type != "rmsnorm":
            raise ValueError("This inference release supports the RMSNorm checkpoint only")
        if config.n_embd < 24 or config.n_layer < 1:
            raise ValueError("Expected n_embd >= 24 and n_layer >= 1")
        dims = tuple(config.matformer_ffn_hidden_dims)
        if len(dims) != 4 or any(d <= 0 for d in dims) or sorted(set(dims)) != list(dims):
            raise ValueError("Expected four increasing positive FFN widths for S/M/L/XL")
        if config.matformer_head_counts and tuple(config.matformer_head_counts) != (config.n_head,) * 4:
            raise ValueError("This release requires fixed attention heads across widths")
        if not config.window_pattern or any(c not in "SL" for c in config.window_pattern.upper()):
            raise ValueError("Expected a nonempty S/L attention window pattern")
        self.config = config
        self.head_dim = get_head_dim(config)
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Preserve padded checkpoint shapes; forward crops logits back to vocab_size.
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "embed_norm": RMSNorm(),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
            "ln_f": RMSNorm(),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Smear: mix previous token's embedding into current token (cheap bigram-like info)
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        # Backout: subtract cached mid-layer residual before final norm to remove low-level features
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        kv_dim = config.n_kv_head * self.head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        self.rotary_seq_len = config.sequence_len
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, self.head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(self.transformer.wte.weight.dtype), sin.to(self.transformer.wte.weight.dtype)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (quarter context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128  # ceil to FA3 tile size (2048 -> 768)
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    @property
    def widths(self):
        return dict(zip(("S", "M", "L", "XL"), self.matformer_ffn_hidden_dims()))

    def new_cache(self, batch_size=1, max_seq_len=None):
        return KVCache(self, batch_size, max_seq_len)

    def get_device(self):
        return self.transformer.wte.weight.device

    def matformer_ffn_hidden_dims(self):
        return tuple(self.transformer.h[0].mlp.ffn_hidden_dims)

    def _resolve_ffn_hidden_dim(self, ffn_hidden_dim=None):
        if isinstance(ffn_hidden_dim, str):
            try:
                ffn_hidden_dim = self.widths[ffn_hidden_dim.upper()]
            except KeyError as exc:
                raise ValueError("Width must be S, M, L, XL, or an integer") from exc
        dims = self.matformer_ffn_hidden_dims()
        max_dim = dims[-1]
        if ffn_hidden_dim is None:
            ffn_hidden_dim = max_dim
        ffn_hidden_dim = int(ffn_hidden_dim)
        if ffn_hidden_dim < 0 or ffn_hidden_dim > max_dim:
            raise ValueError(
                f"ffn_hidden_dim={ffn_hidden_dim} outside [0, {max_dim}]"
            )
        return ffn_hidden_dim

    @torch.inference_mode()
    def forward(self, idx, kv_cache=None, ffn_hidden_dim=None):
        ffn_hidden_dim = self._resolve_ffn_hidden_dim(ffn_hidden_dim)
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        if T == 0:
            raise ValueError("Input must contain at least one token")
        if kv_cache is not None:
            kv_cache.validate(self, idx)
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == self.transformer.wte.weight.dtype, f"Rotary embeddings must be in {self.transformer.wte.weight.dtype}, got {self.cos.dtype}"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        if T0 + T > self.config.sequence_len:
            raise ValueError("Input exceeds the checkpoint context length")
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Embed the tokens
        x = self.transformer.wte(idx) # embed current token
        x = self.transformer.embed_norm(x)

        # Preserve the previous normalized embedding across decode/chunk boundaries.
        previous = None if kv_cache is None else kv_cache.prev_embedding
        shifted = torch.cat(
            [torch.zeros_like(x[:, :1]) if previous is None else previous, x[:, :-1]],
            dim=1,
        )
        if kv_cache is not None:
            kv_cache.prev_embedding = x[:, -1:].clone()
        gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[..., :24]))
        x = x + gate * shifted

        # Forward the trunk of the Transformer
        x0 = x  # save initial normalized embedding for x0 residual
        n_layer = self.config.n_layer
        backout_layer = n_layer // 2  # cache at halfway point
        x_backout = None
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache, ffn_hidden_dim=ffn_hidden_dim)
            if i == backout_layer:
                x_backout = x
        # Subtract mid-layer residual to remove low-level features before logit projection
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = self.transformer.ln_f(x)

        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # fp32 logit softcap
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        return logits
