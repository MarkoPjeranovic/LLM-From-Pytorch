#!/usr/bin/env python3
"""
Pretraining script for CausalLM.

Supports:
  - Multi-GPU via PyTorch DDP (e.g. 2x Kaggle T4s)
  - TensorBoard logging
  - Intermediate checkpoints by step count or wall-clock time
  - Optional BPE tokenizer training from scratch via tokenizers library
  - Resume from checkpoint
  - Optional validation set (--val_dataset_path)
  - Epoch-limited training (--max_epochs)
  - Dataloader inspection (--inspect_dataloader)
  - Gradient checkpointing (--gradient_checkpointing)

Usage (single GPU):
    python training.py --dataset_path /data/corpus --output_dir ./runs/run1

Usage (2 GPUs with torchrun):
    torchrun --nproc_per_node=2 training.py --dataset_path /data/corpus --output_dir ./runs/run1

To train a tokenizer from scratch, pass --train_tokenizer:
    python training.py --train_tokenizer --tokenizer_corpus /data/corpus/*.txt ...
"""

import os
import re
import sys
import time
import math
import argparse
import glob
import hashlib
import struct
from pathlib import Path
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torch.utils.tensorboard.writer import SummaryWriter
import contextlib

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from datasets import Dataset as HFDataset

from config import Config
from model import CausalLM


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

def train_tokenizer(
    corpus_files: list[str],
    vocab_size: int,
    save_path: str,
    eos_token: str = "</s>",
    pad_token: str = "<pad>",
    bos_token: str = "<s>"
) -> Tokenizer:
    """Train a BPE tokenizer from scratch using HuggingFace `tokenizers`."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    special_tokens = [pad_token, eos_token, bos_token]
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
    )
    tokenizer.train(corpus_files, trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    tokenizer.save(save_path)
    print(f"[Tokenizer] Saved to {save_path}  (vocab_size={tokenizer.get_vocab_size()})")
    return tokenizer


def load_tokenizer(path: str) -> Tokenizer:
    return Tokenizer.from_file(path)


# ---------------------------------------------------------------------------
# Document generator  (handles the 2 GB PyArrow limit)
# ---------------------------------------------------------------------------

EOT_MARKER = "<|endoftext|>"
READ_CHUNK = 64 * 1024  # 64 KB readline buffer hint — not used directly but
                         # documents in TinyStories are short so line-by-line is fine


def iter_documents(data_path: str):
    """
    Yield one story/document at a time as a plain string.

    TinyStories files use  <|endoftext|>  as the separator between stories.
    We accumulate lines until we hit that marker, then yield the accumulated
    text (with the marker stripped).  Works for both:
      - a single large .txt file
      - a directory of .txt files
    """
    if os.path.isdir(data_path):
        files = sorted(glob.glob(os.path.join(data_path, "**/*.txt"), recursive=True))
    else:
        files = [data_path]

    assert len(files) > 0, f"No text files found at {data_path}"

    for path in files:
        buf = []
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if EOT_MARKER in line:
                    # Everything before the marker belongs to the current doc
                    before, _, after = line.partition(EOT_MARKER)
                    if before.strip():
                        buf.append(before)
                    if buf:
                        yield "".join(buf)
                    buf = []
                    # Everything after the marker starts the next doc
                    if after.strip():
                        buf.append(after)
                else:
                    buf.append(line)
        # Flush any remaining text at EOF
        if buf:
            text = "".join(buf).strip()
            if text:
                yield text


# ---------------------------------------------------------------------------
# Fast text preprocessing
# ---------------------------------------------------------------------------

# Precompiled pattern — re.sub with a compiled pattern and a fixed replacement
# string is significantly faster than str.replace for large texts because it
# operates on the C side of CPython with no intermediate Python string objects
# for each match.
_EOT_PATTERN    = re.compile(r"<\|endoftext\|>")
_EOT_REPLACEMENT = "</s>"  # just the boundary marker; BOS/EOS added at id level


def preprocess_text(text: str) -> str:
    """Replace <|endoftext|> markers with <s></s> boundary tokens."""
    return _EOT_PATTERN.sub(_EOT_REPLACEMENT, text)


# ---------------------------------------------------------------------------
# Token binary cache  (the fast path)
# ---------------------------------------------------------------------------
# Layout of the .bin file:
#   [8 bytes: uint64 magic=0x544F4B454E530001]
#   [8 bytes: uint64 total token count]
#   [N * 2 bytes: uint16 token ids]
#
# uint16 is sufficient for vocab sizes up to 65535 which covers all common
# BPE tokenizers. Using uint16 instead of int64 gives a 4x size reduction
# and a proportional speedup on the memmap slice + astype path.

_MAGIC  = 0x544F4B454E530001
_HEADER = 16  # bytes


def _bin_cache_path(data_path: str, tokenizer_path: str, cache_dir: str | None) -> str:
    key = hashlib.md5(
        f"{os.path.abspath(data_path)}|{os.path.abspath(tokenizer_path)}".encode()
    ).hexdigest()[:16]
    base = cache_dir if cache_dir else os.path.join(
        os.path.dirname(os.path.abspath(data_path)), ".tok_cache"
    )
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"tokens_{key}.bin")


def _build_bin_cache(
    data_path: str,
    tokenizer: Tokenizer,
    bin_path: str,
    bos_id: int,
    eos_id: int,
) -> int:
    if os.path.isdir(data_path):
        files = sorted(glob.glob(os.path.join(data_path, "**/*.txt"), recursive=True))
    else:
        files = [data_path]
    assert files, f"No text files found at {data_path}"

    total_tokens = 0
    doc_count    = 0
    tmp_path     = bin_path + ".tmp"

    with open(tmp_path, "wb") as out:
        out.write(b"\x00" * _HEADER)

        for file_path in files:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
                remainder = ""
                while True:
                    raw = fh.read(32 * 1024 * 1024)
                    if not raw:
                        break

                    blob  = remainder + raw
                    # Replace <|endoftext|> with </s> — one C-level pass
                    blob  = _EOT_PATTERN.sub(_EOT_REPLACEMENT, blob)

                    # Split on </s>; last piece is an incomplete document tail
                    parts     = blob.split("</s>")
                    remainder = parts[-1]
                    docs      = parts[:-1]

                    for doc in docs:
                        doc = doc.strip()
                        if not doc:
                            continue

                        ids = tokenizer.encode(doc).ids
                        # BOS + content + EOS, entirely at the integer level
                        ids = [bos_id] + ids + [eos_id]

                        np.array(ids, dtype=np.uint16).tofile(out)
                        total_tokens += len(ids)
                        doc_count    += 1

                        if doc_count % 100_000 == 0:
                            print(
                                f"  [Cache] {doc_count:,} docs, "
                                f"{total_tokens:,} tokens …",
                                flush=True,
                            )

                # Flush remainder at EOF
                remainder = remainder.strip()
                if remainder:
                    ids = tokenizer.encode(remainder).ids
                    ids = [bos_id] + ids + [eos_id]
                    np.array(ids, dtype=np.uint16).tofile(out)
                    total_tokens += len(ids)
                    doc_count    += 1

        out.seek(0)
        out.write(struct.pack("<QQ", _MAGIC, total_tokens))

    os.replace(tmp_path, bin_path)
    print(f"[Cache] {doc_count:,} docs → {total_tokens:,} tokens → {bin_path}")
    return total_tokens


def load_or_build_bin_cache(
    data_path: str,
    tokenizer: Tokenizer,
    tokenizer_path: str,
    cache_dir: str | None,
) -> tuple[np.memmap, int]:
    bos_id = tokenizer.token_to_id("<s>")
    eos_id = tokenizer.token_to_id("</s>")

    bin_path = _bin_cache_path(data_path, tokenizer_path, cache_dir)

    if os.path.exists(bin_path):
        with open(bin_path, "rb") as f:
            magic, n_tokens = struct.unpack("<QQ", f.read(_HEADER))
        if magic == _MAGIC:
            print(f"[Cache] Loading token cache: {bin_path}  ({n_tokens:,} tokens)")
        else:
            print(f"[Cache] Cache file corrupt or stale, rebuilding …")
            os.remove(bin_path)
            n_tokens = _build_bin_cache(data_path, tokenizer, bin_path, bos_id, eos_id)
    else:
        n_tokens = _build_bin_cache(data_path, tokenizer, bin_path, bos_id, eos_id)

    tokens = np.memmap(bin_path, dtype=np.uint16, mode="r", offset=_HEADER)
    assert len(tokens) == n_tokens, (
        f"Memmap length {len(tokens):,} != header token count {n_tokens:,}. "
        f"Delete {bin_path} and rerun."
    )
    return tokens, n_tokens


# ---------------------------------------------------------------------------
# Build HF Dataset from memmap  (zero-copy slice view)
# ---------------------------------------------------------------------------

def build_hf_dataset(
    data_path: str,
    tokenizer: Tokenizer,
    tokenizer_path: str,
    seq_len: int,
    rank: int = 0,
    world_size: int = 1,
    cache_dir: str | None = None,
) -> HFDataset:
    """
    Return a HuggingFace Dataset backed by the token memmap.

    The dataset has a single column "chunk_idx" (the starting token index for
    each sample). input_ids and labels are produced on-the-fly in
    make_batch_iter() by slicing the memmap, so there is no duplication of
    token data on disk or in RAM.

    Sharding for DDP is done by assigning every world_size-th chunk to this
    rank (interleaved, not contiguous, so each rank sees a representative
    mix of the data).
    """
    tokens, n_tokens = load_or_build_bin_cache(
        data_path, tokenizer, tokenizer_path, cache_dir
    )

    chunk_size  = seq_len + 1
    # Total number of complete chunks in the token stream
    n_chunks    = (n_tokens - 1) // chunk_size

    # Chunk start indices for this rank
    all_starts  = np.arange(n_chunks, dtype=np.int64) * chunk_size
    rank_starts = all_starts[rank::world_size]

    dataset = HFDataset.from_dict({"chunk_idx": rank_starts.tolist()})
    # Attach the memmap as a custom attribute so make_batch_iter can reach it
    dataset._tok_memmap  = tokens
    dataset._seq_len     = seq_len

    return dataset


# ---------------------------------------------------------------------------
# Batch iterator  (slices the memmap directly — no HF arrow involved)
# ---------------------------------------------------------------------------

def make_batch_iter(
    dataset: HFDataset,
    batch_size: int,
    shuffle: bool,
    seed: int = 0,
):
    """
    Yield (input_ids, labels) int64 tensors of shape [batch_size, seq_len].

    Slicing a numpy memmap returns a view (no copy) until we call .astype()
    to promote uint16 → int64.  That promotion is unavoidable because
    PyTorch embedding layers require int64, but it happens on a small
    (batch_size × seq_len) array, not the whole token file.

    Label shifting (labels = input_ids shifted left by one) is handled here
    rather than in the dataset so we only store one copy of the tokens.
    The model receives pre-shifted labels so it does not need to do it.
    """
    tokens  = dataset._tok_memmap
    seq_len = dataset._seq_len

    indices = np.arange(len(dataset), dtype=np.int64)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for batch_start in range(0, len(indices) - batch_size + 1, batch_size):
        batch_indices = indices[batch_start: batch_start + batch_size]
        starts        = dataset["chunk_idx"][batch_indices]  # list of ints

        # Stack slices into [batch_size, seq_len+1]
        # Each slice is a uint16 memmap view; astype copies to int64 once.
        rows = np.stack([
            np.array(tokens[s: s + seq_len + 1], dtype=np.int64)
            for s in starts
        ])

        x = torch.from_numpy(rows[:, :-1])  # [B, seq_len]
        y = torch.from_numpy(rows[:, 1:])   # [B, seq_len]  — shifted labels
        yield x, y


# ---------------------------------------------------------------------------
# Dataloader inspection
# ---------------------------------------------------------------------------

def inspect_hf_dataset(
    dataset: HFDataset,
    tokenizer: Tokenizer,
    output_dir: str,
    n_items: int = 100,
):
    out_path = os.path.join(output_dir, "dataloader_inspect.txt")
    n_items  = min(n_items, len(dataset))
    tokens   = dataset._tok_memmap
    seq_len  = dataset._seq_len

    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(n_items):
            start = dataset["chunk_idx"][i]
            ids   = np.array(tokens[start: start + seq_len], dtype=np.int64).tolist()

            token_strs = [tokenizer.id_to_token(t) or "<None>" for t in ids[:32]]
            decoded    = tokenizer.decode(ids, skip_special_tokens=False)

            f.write(f"=== Sample {i} ===\n")
            f.write(f"Raw IDs : {ids[:32]}{' ...' if len(ids) > 32 else ''}\n")
            f.write(f"Tokens  : {token_strs}{' ...' if len(ids) > 32 else ''}\n")
            f.write(f"Decoded : {decoded}\n\n")

    print(f"[Inspect] Wrote {n_items} samples to {out_path}")


# ---------------------------------------------------------------------------
# OneBatchLoader
# ---------------------------------------------------------------------------

class OneBatchLoader:
    def __init__(self, dataset: HFDataset, batch_size: int, device=None):
        gen  = make_batch_iter(dataset, batch_size, shuffle=False)
        self.x, self.y = next(gen)
        if device is not None:
            self.x = self.x.to(device)
            self.y = self.y.to(device)

    def __iter__(self):
        return self

    def __next__(self):
        return self.x, self.y


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_validation(
    model, val_dataset: HFDataset, batch_size: int, device, use_amp: bool
):
    model.eval()
    total_loss    = 0.0
    total_batches = 0

    for x, y in make_batch_iter(val_dataset, batch_size, shuffle=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with autocast("cuda", enabled=use_amp, dtype=torch.float16):
            _, loss = model(
                input_ids=x,
                attention_mask=None,
                start_pos=0,
                cache_k=None,
                cache_v=None,
                labels=y,
            )
        total_loss    += loss.item()
        total_batches += 1

    model.train()
    return total_loss / total_batches if total_batches > 0 else float("nan")


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------

def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, step, config, path):
    raw_model = model.module if isinstance(model, DDP) else model
    torch.save(
        {
            "model_state_dict":     raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step":                 step,
            "config":               asdict(config),
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None):
    ckpt      = torch.load(path, map_location="cpu", weights_only=False)
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("step", 0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pretrain CausalLM")

    # Data / tokenizer
    parser.add_argument("--dataset_path",     type=str, required=True)
    parser.add_argument("--val_dataset_path", type=str, default=None)
    parser.add_argument("--tokenizer_path",   type=str, default="tokenizer.json")
    parser.add_argument("--train_tokenizer",  action="store_true")
    parser.add_argument("--tokenizer_corpus", type=str, default=None)
    parser.add_argument("--hf_cache_dir",     type=str, default=None,
                        help="Root directory for Arrow dataset caches")

    # Model
    parser.add_argument("--hidden_size",             type=int, default=480)
    parser.add_argument("--num_hidden_layers",       type=int, default=16)
    parser.add_argument("--num_attention_heads",     type=int, default=8)
    parser.add_argument("--num_key_value_heads",     type=int, default=2)
    parser.add_argument("--intermediate_size",       type=int, default=1280)
    parser.add_argument("--max_position_embeddings", type=int, default=2048)
    parser.add_argument("--vocab_size",              type=int, default=16384)

    # Training
    parser.add_argument("--seq_len",          type=int,   default=1024)
    parser.add_argument("--batch_size",       type=int,   default=4)
    parser.add_argument("--grad_accum_steps", type=int,   default=8)
    parser.add_argument("--max_steps",        type=int,   default=1000000)
    parser.add_argument("--max_epochs",       type=int,   default=0)
    parser.add_argument("--warmup_steps",     type=int,   default=2000)
    parser.add_argument("--max_lr",           type=float, default=3e-4)
    parser.add_argument("--min_lr",           type=float, default=3e-5)
    parser.add_argument("--weight_decay",     type=float, default=0.1)
    parser.add_argument("--grad_clip",        type=float, default=1.0)
    parser.add_argument("--use_amp",          action="store_true", default=False)

    # Memory optimisation
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Recompute activations during backward to save GPU memory "
                             "at the cost of ~30%% extra compute. Requires model support "
                             "(see enable_gradient_checkpointing() in model.py).")

    # Checkpointing
    parser.add_argument("--output_dir",         type=str, default="./runs/run1")
    parser.add_argument("--save_every_steps",   type=int, default=5000)
    parser.add_argument("--save_every_seconds", type=int, default=0)
    parser.add_argument("--resume_from",        type=str, default=None)

    # Misc
    parser.add_argument("--log_every",          type=int,  default=10)
    parser.add_argument("--inspect_dataloader", action="store_true")
    parser.add_argument("--overfit_one_batch",  action="store_true")

    args = parser.parse_args()

    # ---- DDP setup --------------------------------------------------------
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group(backend="nccl")
        rank       = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        device     = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        rank       = 0
        local_rank = 0
        world_size = 1
        device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_master = rank == 0
    if is_master:
        os.makedirs(args.output_dir, exist_ok=True)

    # ---- Tokenizer --------------------------------------------------------
    if args.train_tokenizer and is_master:
        corpus_pattern = args.tokenizer_corpus or os.path.join(args.dataset_path, "**/*.txt")
        corpus_files   = sorted(glob.glob(corpus_pattern, recursive=True))
        if not corpus_files and os.path.isfile(args.dataset_path):
            corpus_files = [args.dataset_path]
        assert corpus_files, "No corpus files found for tokenizer training"
        tokenizer = train_tokenizer(
            corpus_files,
            vocab_size=args.vocab_size,
            save_path=os.path.join(args.output_dir, "tokenizer.json"),
        )
        args.tokenizer_path = os.path.join(args.output_dir, "tokenizer.json")
        args.vocab_size     = tokenizer.get_vocab_size()

    if ddp:
        dist.barrier()

    tokenizer       = load_tokenizer(args.tokenizer_path)
    args.vocab_size = tokenizer.get_vocab_size()
    padded_vocab    = math.ceil(args.vocab_size / 64) * 64

    # ---- Config / Model ---------------------------------------------------
    config = Config(
        vocab_size=padded_vocab,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        max_position_embeddings=args.max_position_embeddings,
    )

    model = CausalLM(config)
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
        if is_master:
            print("[Model] Gradient checkpointing enabled")

    model.init_weights()
    model = model.to(device)

    if is_master:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[Model] {n_params/1e6:.1f}M parameters  |  config: {config}")

    if ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # ---- Optimizer --------------------------------------------------------
    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() < 2 or "norm" in name or "bias" in name:
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.max_lr,
        betas=(0.9, 0.95),
        fused=torch.cuda.is_available(),
    )

    scaler = GradScaler("cuda", enabled=args.use_amp)

    # ---- Resume -----------------------------------------------------------
    start_step = 0
    if args.resume_from and os.path.exists(args.resume_from):
        start_step = load_checkpoint(args.resume_from, model, optimizer)
        if is_master:
            print(f"[Resume] Loaded checkpoint, step={start_step}")

    # ---- Datasets ---------------------------------------------------------
    # Master builds the binary token cache first, then all ranks load their
    # own shard from the guaranteed-existing cache.
    if is_master:
        load_or_build_bin_cache(
            args.dataset_path, tokenizer, args.tokenizer_path, args.hf_cache_dir
        )

    if ddp:
        dist.barrier()

    train_dataset = build_hf_dataset(
        args.dataset_path, tokenizer, args.tokenizer_path,
        args.seq_len, rank, world_size, args.hf_cache_dir,
    )
    if is_master:
        print(f"[Dataset] Training samples (this rank): {len(train_dataset)}")

    # Inspect and exit early if requested
    if args.inspect_dataloader:
        if is_master:
            inspect_hf_dataset(train_dataset, tokenizer, args.output_dir, n_items=100)
        if ddp:
            dist.barrier()
            dist.destroy_process_group()
        return

    val_dataset = None
    if args.val_dataset_path is not None:
        if is_master:
            load_or_build_bin_cache(
                args.val_dataset_path, tokenizer, args.tokenizer_path, args.hf_cache_dir
            )
        if ddp:
            dist.barrier()
        val_dataset = build_hf_dataset(
            args.val_dataset_path, tokenizer, args.tokenizer_path,
            args.seq_len, rank, world_size, args.hf_cache_dir,
        )
        if is_master:
            print(f"[Dataset] Validation samples (this rank): {len(val_dataset)}")

    # ---- LR schedule ------------------------------------------------------
    # Compute total optimizer steps based on whichever limit is active so that
    # the cosine decay reaches min_lr exactly at the end of training whether
    # that is determined by max_steps or max_epochs.
    steps_per_epoch           = len(train_dataset) // args.batch_size
    optimizer_steps_per_epoch = steps_per_epoch // args.grad_accum_steps

    if args.max_epochs > 0:
        total_optimizer_steps = optimizer_steps_per_epoch * args.max_epochs
    else:
        total_optimizer_steps = args.max_steps // args.grad_accum_steps

    # ---- TensorBoard ------------------------------------------------------
    writer = None
    if is_master:
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tb"))

    # ---- Training loop ----------------------------------------------------
    model.train()

    if args.overfit_one_batch and is_master:
        print("[Mode] Overfitting single batch")

    epoch          = 0
    global_step    = start_step
    accum_loss     = 0.0
    last_ckpt_time = time.time()

    if is_master:
        print(f"[Train] step={global_step}, total_optimizer_steps={total_optimizer_steps}, "
              f"effective_batch={args.batch_size * args.grad_accum_steps * world_size}")
        if args.max_epochs > 0:
            print(f"[Train] max_epochs={args.max_epochs}")

    def fresh_iter(ep: int):
        if args.overfit_one_batch:
            return iter(OneBatchLoader(train_dataset, args.batch_size, device))
        return make_batch_iter(
            train_dataset, args.batch_size,
            shuffle=True, seed=ep * world_size + rank,
        )

    data_iter = fresh_iter(epoch)

    while global_step < args.max_steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            # End of epoch — run validation then decide whether to continue.
            if val_dataset is not None and is_master:
                vl      = run_validation(model, val_dataset, args.batch_size, device, args.use_amp)
                opt_step = global_step // args.grad_accum_steps
                print(f"[Val] epoch={epoch} step={opt_step} val_loss={vl:.4f}")
                if writer:
                    writer.add_scalar("val/loss", vl, opt_step)

            epoch += 1
            if args.max_epochs > 0 and epoch >= args.max_epochs:
                if is_master:
                    print(f"[Train] max_epochs={args.max_epochs} reached.")
                break

            data_iter = fresh_iter(epoch)
            x, y     = next(data_iter)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Only sync gradients on the last micro-step of each accumulation window.
        sync_ctx = (
            model.no_sync()
            if ddp and (global_step + 1) % args.grad_accum_steps != 0
            else contextlib.nullcontext()
        )

        with sync_ctx:
            with autocast("cuda", enabled=args.use_amp, dtype=torch.float16):
                _, loss = model(
                    input_ids=x, attention_mask=None,
                    start_pos=0, cache_k=None, cache_v=None,
                    labels=y,
                )
                loss = loss / args.grad_accum_steps
            scaler.scale(loss).backward()
            accum_loss += loss.item()

        if (global_step + 1) % args.grad_accum_steps == 0:
            opt_step = (global_step + 1) // args.grad_accum_steps

            lr = get_lr(opt_step, args.warmup_steps, total_optimizer_steps,
                        args.max_lr, args.min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            step_loss  = accum_loss
            accum_loss = 0.0

            if is_master and opt_step % args.log_every == 0:
                print(f"step={opt_step}  loss={step_loss:.4f}  lr={lr:.2e}")
                if writer:
                    writer.add_scalar("train/loss", step_loss, opt_step)
                    writer.add_scalar("train/lr",   lr,        opt_step)

            if is_master and args.save_every_steps > 0 and opt_step % args.save_every_steps == 0:
                p = os.path.join(args.output_dir, f"ckpt_step{opt_step}.pt")
                if val_dataset is not None:
                    vl = run_validation(model, val_dataset, args.batch_size, device, args.use_amp)
                    print(f"[Val] step={opt_step} val_loss={vl:.4f}")
                    if writer:
                        writer.add_scalar("val/loss", vl, opt_step)
                save_checkpoint(model, optimizer, global_step, config, p)
                print(f"[Ckpt] {p}")

            if is_master and args.save_every_seconds > 0:
                now = time.time()
                if now - last_ckpt_time >= args.save_every_seconds:
                    p = os.path.join(args.output_dir, f"ckpt_time_{int(now)}.pt")
                    if val_dataset is not None:
                        vl = run_validation(model, val_dataset, args.batch_size, device, args.use_amp)
                        print(f"[Val] step={opt_step} val_loss={vl:.4f}")
                        if writer:
                            writer.add_scalar("val/loss", vl, opt_step)
                    save_checkpoint(model, optimizer, global_step, config, p)
                    print(f"[Ckpt] timed {p}")
                    last_ckpt_time = now

        global_step += 1

    # ---- Final ------------------------------------------------------------
    if is_master:
        if val_dataset is not None:
            vl = run_validation(model, val_dataset, args.batch_size, device, args.use_amp)
            print(f"[Val] final val_loss={vl:.4f}")
            if writer:
                writer.add_scalar("val/loss", vl, global_step // args.grad_accum_steps)

        p = os.path.join(args.output_dir, "ckpt_final.pt")
        save_checkpoint(model, optimizer, global_step, config, p)
        print(f"[Done] {p}")

        if writer:
            writer.close()

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()