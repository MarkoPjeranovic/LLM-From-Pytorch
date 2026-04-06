#!/usr/bin/env python3
"""
Inference script for CausalLM.

Supports:
  - Greedy / sampling generation
  - Modern samplers: temperature, top_k, top_p (nucleus), min_p
  - With and without KV cache
  - Repetition penalty

Usage:
    python inference.py \
        --checkpoint ./runs/run1/ckpt_final.pt \
        --tokenizer_path ./runs/run1/tokenizer.json \
        --prompt "Once upon a time" \
        --max_new_tokens 256 \
        --temperature 0.8 \
        --min_p 0.05

    # Without KV cache (slower, useful for debugging / comparison):
    python inference.py --checkpoint ... --no_kv_cache --prompt "Hello"
"""

import argparse
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from config import Config
from model import CausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Sampling utilities
# ---------------------------------------------------------------------------

def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Scale logits by temperature. temperature=0 -> greedy (handled outside)."""
    if temperature <= 0:
        return logits
    return logits / temperature


def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Zero-out all logits outside the top-k."""
    if k <= 0 or k >= logits.size(-1):
        return logits
    top_k_vals, _ = torch.topk(logits, k, dim=-1)
    threshold = top_k_vals[..., -1:]
    logits = logits.masked_fill(logits < threshold, float("-inf"))
    return logits


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus sampling: keep smallest set of tokens whose cumulative prob >= p."""
    if p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens with cumulative prob above the threshold (keep at least 1)
    sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= p
    sorted_logits[sorted_mask] = float("-inf")
    # Scatter back
    logits = logits.scatter(-1, sorted_indices, sorted_logits)
    return logits


def apply_min_p(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    """
    min_p sampling: discard tokens whose probability is less than
    min_p * max_probability.  See https://arxiv.org/abs/2407.01082
    """
    if min_p <= 0.0:
        return logits
    probs = F.softmax(logits, dim=-1)
    max_prob = probs.max(dim=-1, keepdim=True).values
    threshold = min_p * max_prob
    logits = logits.masked_fill(probs < threshold, float("-inf"))
    return logits


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: list[int],
    penalty: float,
) -> torch.Tensor:
    """Penalise tokens that have already been generated."""
    if penalty == 1.0 or len(generated_ids) == 0:
        return logits
    unique_ids = list(set(generated_ids))
    score = logits[..., unique_ids]
    # If score > 0 divide by penalty, else multiply by penalty
    score = torch.where(score > 0, score / penalty, score * penalty)
    logits[..., unique_ids] = score
    return logits


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    generated_ids: list[int] | None = None,
) -> int:
    """Apply samplers in order and return a single token id."""
    logits = logits.clone()

    # Repetition penalty first (operates on raw logits)
    if generated_ids:
        logits = apply_repetition_penalty(logits, generated_ids, repetition_penalty)

    # Temperature
    if temperature <= 0:
        # Greedy
        return int(logits.argmax(dim=-1).item())

    logits = apply_temperature(logits, temperature)

    # min_p (before top-k/top-p so the threshold is meaningful)
    logits = apply_min_p(logits, min_p)

    # top-k
    logits = apply_top_k(logits, top_k)

    # top-p / nucleus
    logits = apply_top_p(logits, top_p)

    probs = F.softmax(logits, dim=-1)
    token_id = int(torch.multinomial(probs, num_samples=1).item())
    return token_id

def debug_top_tokens(
    logits: torch.Tensor,
    tokenizer: Tokenizer,
    k: int = 10,
    prefix: str = "",
):
    """
    Print top-k tokens and probabilities for debugging.
    logits: (vocab,)
    """
    probs = F.softmax(logits, dim=-1)

    top_probs, top_ids = torch.topk(probs, k)

    print(f"\n{prefix}Top {k} tokens:")
    for i in range(k):
        token_id = int(top_ids[i])
        prob = float(top_probs[i])

        # Decode safely (handles weird tokens)
        try:
            token_str = tokenizer.decode([token_id])
        except:
            token_str = "<decode_error>"

        token_str = token_str.replace("\n", "\\n")

        print(f"{i:2d}: id={token_id:<6} prob={prob:.4f} token='{token_str}'")

# ---------------------------------------------------------------------------
# KV-cache helpers
# ---------------------------------------------------------------------------

def make_kv_cache(config: Config, batch_size: int, max_len: int, device: torch.device, dtype: torch.dtype):
    """Allocate KV caches: list of (cache_k, cache_v) per layer."""
    head_dim = config.hidden_size // config.num_attention_heads
    cache_k = torch.zeros(
        config.num_hidden_layers, batch_size, config.num_key_value_heads, max_len, head_dim,
        device=device, dtype=dtype,
    )
    cache_v = torch.zeros(
        config.num_hidden_layers, batch_size, config.num_key_value_heads, max_len, head_dim,
        device=device, dtype=dtype,
    )
    return cache_k, cache_v

def make_causal_mask(seq_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Generation with KV cache
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_with_cache(
    model: CausalLM,
    input_ids: torch.Tensor,  # (1, prompt_len)
    max_new_tokens: int,
    config: Config,
    eos_token_id: int | None = None,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
) -> list[int]:
    device = input_ids.device
    dtype = next(model.parameters()).dtype
    prompt_len = input_ids.shape[1]
    total_len = prompt_len + max_new_tokens

    cache_k, cache_v = make_kv_cache(config, 1, total_len, device, dtype)

    # Prefill: process entire prompt at once
    causal_mask = make_causal_mask(prompt_len, torch.float32, device)
    logits, _ = model(
        input_ids=input_ids,
        attention_mask=causal_mask,
        start_pos=0,
        cache_k=cache_k,
        cache_v=cache_v,
    )
    # logits: (1, prompt_len, vocab_size)
    next_logits = logits[:, -1, :]  # (1, vocab_size)

    generated_ids = input_ids[0].tolist()

    for i in range(max_new_tokens):
        token_id = sample_token(
            next_logits[0],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            generated_ids=generated_ids,
        )
        generated_ids.append(token_id)

        if eos_token_id is not None and token_id == eos_token_id:
            break

        # Decode step: single token, no causal mask needed (seq_len=1)
        new_input = torch.tensor([[token_id]], device=device, dtype=input_ids.dtype)
        logits, _ = model(
            input_ids=new_input,
            attention_mask=None,  # seq_len=1, no masking needed
            start_pos=prompt_len + i,
            cache_k=cache_k,
            cache_v=cache_v,
        )
        next_logits = logits[:, -1, :]

    return generated_ids


# ---------------------------------------------------------------------------
# Generation without KV cache (recompute full sequence each step)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_no_cache(
    model: CausalLM,
    input_ids: torch.Tensor,  # (1, prompt_len)
    max_new_tokens: int,
    config: Config,
    eos_token_id: int | None = None,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
) -> list[int]:
    device = input_ids.device
    generated_ids = input_ids[0].tolist()

    for _ in range(max_new_tokens):
        seq = torch.tensor([generated_ids], device=device, dtype=input_ids.dtype)
        seq_len = seq.shape[1]
        causal_mask = make_causal_mask(seq_len, torch.float32, device)

        logits, _ = model(
            input_ids=seq,
            attention_mask=causal_mask,
            start_pos=0,
            cache_k=None,
            cache_v=None,
        )
        next_logits = logits[:, -1, :]

        token_id = sample_token(
            next_logits[0],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            generated_ids=generated_ids,
        )
        generated_ids.append(token_id)

        if eos_token_id is not None and token_id == eos_token_id:
            break

    return generated_ids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Inference with CausalLM")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer.json")
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--no_kv_cache", action="store_true", help="Disable KV cache (recompute each step)")
    # Sampling
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--min_p", type=float, default=0.05)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    device = torch.device(args.device)

    # ---- Load checkpoint --------------------------------------------------
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config_dict = ckpt["config"]
    config = Config(**config_dict)

    model = CausalLM(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    print(f"[Model] Loaded from {args.checkpoint}  ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")

    # ---- Tokenizer --------------------------------------------------------
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    eos_id = config.eos_token_id

    # ---- Encode prompt ----------------------------------------------------
    encoded = tokenizer.encode(args.prompt)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    print(f"[Prompt] \"{args.prompt}\"  ({input_ids.shape[1]} tokens)")

    # ---- Generate ---------------------------------------------------------
    t0 = time.time()

    gen_fn = generate_no_cache if args.no_kv_cache else generate_with_cache
    output_ids = gen_fn(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        config=config,
        eos_token_id=eos_id,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
    )

    elapsed = time.time() - t0
    new_tokens = len(output_ids) - input_ids.shape[1]

    # ---- Decode & print ---------------------------------------------------
    text = tokenizer.decode(output_ids)
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)
    print(f"\n[Stats] {new_tokens} tokens in {elapsed:.2f}s  ({new_tokens/elapsed:.1f} tok/s)")
    print(f"[Mode]  {'No KV cache' if args.no_kv_cache else 'KV cache'}")
    print(f"[Samplers] temp={args.temperature} top_k={args.top_k} top_p={args.top_p} "
          f"min_p={args.min_p} rep_penalty={args.repetition_penalty}")


if __name__ == "__main__":
    main()