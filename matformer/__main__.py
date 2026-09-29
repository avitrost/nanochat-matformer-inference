import argparse

import torch

from . import generate, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Minimal MatFormer text completion")
    parser.add_argument("--prompt", default="The future of artificial intelligence is")
    parser.add_argument("--width", choices=["S", "M", "L", "XL"], default="XL")
    parser.add_argument("--switch-to", choices=["S", "M", "L", "XL"])
    parser.add_argument("--switch-after", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()
    if args.switch_after < 0:
        parser.error("--switch-after must be nonnegative")
    model, tokenizer, _ = load_checkpoint(
        device=args.device, dtype=getattr(torch, args.dtype) if args.dtype else None)
    bos = tokenizer.encode_single_token("<|bos|>")
    tokens = [bos] + tokenizer.encode_ordinary(args.prompt)
    width = args.width
    if args.switch_to:
        width = lambda step, ids: args.width if step < args.switch_after else args.switch_to
    outputs = list(generate(model, tokens, args.max_new_tokens, width, args.temperature,
                            args.seed, stop_token_id=bos))
    if outputs and outputs[-1] == bos:
        outputs.pop()
    print(args.prompt + tokenizer.decode(outputs))


if __name__ == "__main__":
    main()
