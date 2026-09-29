"""Single-sequence greedy or sampled decoding with a per-step width policy."""

import torch


@torch.inference_mode()
def generate(model, tokens, max_new_tokens=64, width="XL", temperature=0.0, seed=42,
             stop_token_id=None):
    """Yield token IDs. width is a label/int or callable(step, token_ids).

    Step 0 selects the prefill width and predicts the first output token.
    Later steps process the previous output token using the selected width.
    Every step reuses the same cache; token_ids includes the prompt and outputs.
    """
    tokens = list(tokens)
    if not tokens or max_new_tokens < 0 or temperature < 0:
        raise ValueError("Expected a nonempty prompt and nonnegative generation settings")
    if len(tokens) + max(0, max_new_tokens - 1) > model.config.sequence_len:
        raise ValueError("Prompt plus decoding inputs exceeds the checkpoint context length")
    if max_new_tokens == 0:
        return
    cache = model.new_cache(max_seq_len=len(tokens) + max_new_tokens - 1)
    device = model.get_device()
    sample_device = torch.device("cpu") if device.type == "mps" else device
    rng = torch.Generator(device=sample_device).manual_seed(seed) if temperature else None
    inputs = torch.tensor([tokens], device=device)
    for step in range(max_new_tokens):
        active_width = width(step, tuple(tokens)) if callable(width) else width
        logits = model(inputs, kv_cache=cache, ffn_hidden_dim=active_width)[:, -1]
        if temperature:
            probabilities = (logits / temperature).softmax(-1).to(sample_device)
            token = int(torch.multinomial(probabilities, 1, generator=rng))
        else:
            token = int(logits.argmax(-1))
        tokens.append(token)
        yield token
        if token == stop_token_id:
            break
        inputs = torch.tensor([[token]], device=device)
