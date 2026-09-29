"""Fixed-attention KV cache; FFN width is deliberately not part of its shape."""

import torch


class KVCache:
    def __init__(self, model, batch_size=1, max_seq_len=None):
        config = model.config
        self.max_seq_len = config.sequence_len if max_seq_len is None else max_seq_len
        if batch_size < 1 or not 1 <= self.max_seq_len <= config.sequence_len:
            raise ValueError("Invalid batch size or cache capacity")
        self.n_layers = config.n_layer
        self.batch_size = batch_size
        self._model = model
        weight = model.transformer.wte.weight
        shape = (config.n_layer, batch_size, self.max_seq_len, config.n_kv_head, model.head_dim)
        self.k_cache = torch.empty(shape, device=weight.device, dtype=weight.dtype)
        self.v_cache = torch.empty_like(self.k_cache)
        self.cache_seqlens = torch.zeros(batch_size, device=weight.device, dtype=torch.int32)
        self.prev_embedding = None

    def get_pos(self):
        return int(self.cache_seqlens[0])

    def get_layer_cache(self, layer_idx):
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def advance(self, num_tokens):
        self.cache_seqlens += num_tokens

    def reset(self):
        self.cache_seqlens.zero_()
        self.prev_embedding = None

    def validate(self, model, idx):
        weight = model.transformer.wte.weight
        if self._model is not model:
            raise ValueError("Cache belongs to a different model")
        if idx.size(0) != self.batch_size:
            raise ValueError("Cache batch size does not match input")
        if self.k_cache.device != weight.device or self.k_cache.dtype != weight.dtype:
            raise ValueError("Create a new cache after moving or casting the model")
        if not bool(torch.all(self.cache_seqlens == self.cache_seqlens[0])):
            raise ValueError("Only equal-length batches are supported")
        if self.get_pos() + idx.size(1) > self.max_seq_len:
            raise ValueError("KV cache capacity exceeded")
