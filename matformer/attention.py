"""PyTorch SDPA attention with bottom-right causal alignment for cached chunks."""

import torch
import torch.nn.functional as F


def _attention(q, k, v, window_size):
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))
    tq, tk = q.size(2), k.size(2)
    window = window_size[0]
    gqa = q.size(1) != k.size(1)
    if tq == tk and (window < 0 or window >= tq):
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=gqa)
    elif tq == 1:
        start = 0 if window < 0 else max(0, tk - window - 1)
        y = F.scaled_dot_product_attention(q, k[:, :, start:], v[:, :, start:], enable_gqa=gqa)
    else:
        rows = tk - tq + torch.arange(tq, device=q.device)[:, None]
        cols = torch.arange(tk, device=q.device)[None, :]
        mask = cols <= rows
        if window >= 0:
            mask &= rows - cols <= window
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=gqa)
    return y.transpose(1, 2)


def flash_attn_func(q, k, v, causal=True, window_size=(-1, 0)):
    return _attention(q, k, v, window_size)


def flash_attn_with_kvcache(q, k_cache, v_cache, *, k, v, cache_seqlens,
                          causal=True, window_size=(-1, 0)):
    pos = int(cache_seqlens[0])
    end = pos + q.size(1)
    k_cache[:, pos:end] = k
    v_cache[:, pos:end] = v
    return _attention(q, k_cache[:, :end], v_cache[:, :end], window_size)
