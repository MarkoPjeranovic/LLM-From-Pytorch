#!/usr/bin/env python3
"""
Pretraining script for CausalLM.

Supports:
  - Multi-GPU via PyTorch DDP (e.g. 2x Kaggle T4s)
  - TensorBoard logging
  - Intermediate checkpoints by step count or wall-clock time
  - Optional BPE tokenizer training from scratch via tokenizers library
  - Resume from checkpoint

Usage (single GPU):
    python training.py --dataset_path /data/corpus --output_dir ./runs/run1

Usage (2 GPUs with torchrun):
    torchrun --nproc_per_node=2 training.py --dataset_path /data/corpus --output_dir ./runs/run1

To train a tokenizer from scratch, pass --train_tokenizer:
    python training.py --train_tokenizer --tokenizer_corpus /data/corpus/*.txt ...
"""

import os
import sys
import time
import math
import argparse
import glob
from pathlib import Path
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torch.utils.tensorboard.writer import SummaryWriter
import contextlib

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

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
# Dataset
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    """
    Memory-maps a directory of .txt (or already-tokenized .bin) files and
    returns chunks of `seq_len + 1` token ids (input + label shift by 1).
    
    For simplicity this version reads all text, tokenizes into one giant
    1-D array, then serves fixed-length windows.  For very large corpora
    consider memory-mapped numpy arrays or streaming.
    """

    def __init__(self, data_path: str, tokenizer: Tokenizer, seq_len: int):
        super().__init__()
        self.seq_len = seq_len

        bin_cache = data_path.rstrip("/") + ".tokens.bin"
        if os.path.exists(bin_cache):
            print(f"[Dataset] Loading cached tokens from {bin_cache}")
            self.tokens = torch.load(bin_cache, weights_only=True)
        else:
            # Gather text files
            if os.path.isdir(data_path):
                files = sorted(glob.glob(os.path.join(data_path, "**/*.txt"), recursive=True))
            else:
                files = [data_path]
            assert len(files) > 0, f"No .txt files found in {data_path}"

            all_ids: list[int] = []
            for f in files:
                with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                    text = fh.read()
                encoded = tokenizer.encode(text)
                all_ids.append(tokenizer.token_to_id("<s>"))
                all_ids.extend(encoded.ids)
                all_ids.append(tokenizer.token_to_id("</s>"))
            self.tokens = torch.tensor(all_ids, dtype=torch.long)
            torch.save(self.tokens, bin_cache)
            print(f"[Dataset] Tokenized {len(files)} file(s) -> {len(self.tokens)} tokens, cached to {bin_cache}")

        # Number of full windows we can serve
        self.n_samples = (len(self.tokens) - 1) // self.seq_len

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.tokens[start : start + self.seq_len + 1]
        x = chunk[:-1]  # input
        y = chunk[1:]    # labels
        return x, y


# ---------------------------------------------------------------------------
# Learning-rate schedule (cosine with warmup)
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
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "config": asdict(config),
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("step", 0)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pretrain CausalLM")
    # Data / tokenizer
    parser.add_argument("--dataset_path", type=str, required=True, help="Dir or file of .txt training data")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer.json", help="Path to tokenizer json")
    parser.add_argument("--train_tokenizer", action="store_true", help="Train tokenizer from scratch")
    parser.add_argument("--tokenizer_corpus", type=str, default=None,
                        help="Glob pattern for tokenizer training corpus (defaults to dataset_path)")
    # Model size overrides (small defaults for quick iteration)
    parser.add_argument("--hidden_size", type=int, default=960)
    parser.add_argument("--num_hidden_layers", type=int, default=32)
    parser.add_argument("--num_attention_heads", type=int, default=16)
    parser.add_argument("--num_key_value_heads", type=int, default=4)
    parser.add_argument("--intermediate_size", type=int, default=2560)
    parser.add_argument("--max_position_embeddings", type=int, default=2048)
    parser.add_argument("--vocab_size", type=int, default=50304)
    # Training hyper-parameters
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4, help="Per-GPU micro batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=100_000)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--use_amp", action="store_true", default=True)
    # Checkpointing
    parser.add_argument("--output_dir", type=str, default="./runs/run1")
    parser.add_argument("--save_every_steps", type=int, default=5000, help="Save checkpoint every N steps")
    parser.add_argument("--save_every_seconds", type=int, default=0,
                        help="Save checkpoint every N seconds (0=disabled)")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")
    # Logging
    parser.add_argument("--log_every", type=int, default=10)

    args = parser.parse_args()

    # ---- DDP setup --------------------------------------------------------
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_master = rank == 0
    if is_master:
        os.makedirs(args.output_dir, exist_ok=True)

    # ---- Tokenizer --------------------------------------------------------
    if args.train_tokenizer and is_master:
        corpus_pattern = args.tokenizer_corpus or os.path.join(args.dataset_path, "**/*.txt")
        corpus_files = sorted(glob.glob(corpus_pattern, recursive=True))
        if not corpus_files and os.path.isfile(args.dataset_path):
            corpus_files = [args.dataset_path]
        assert len(corpus_files) > 0, "No corpus files found for tokenizer training"
        tokenizer = train_tokenizer(
            corpus_files,
            vocab_size=args.vocab_size,
            save_path=os.path.join(args.output_dir, "tokenizer.json"),
        )
        args.tokenizer_path = os.path.join(args.output_dir, "tokenizer.json")
        # Update vocab size to match trained tokenizer
        args.vocab_size = tokenizer.get_vocab_size()

    if ddp:
        dist.barrier()  # wait for tokenizer training

    tokenizer = load_tokenizer(args.tokenizer_path)
    args.vocab_size = tokenizer.get_vocab_size()
    # Pad to multiple of 64 for tensor-core efficiency
    padded_vocab = math.ceil(args.vocab_size / 64) * 64

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
    model.init_weights()
    model = model.to(device)

    if is_master:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[Model] {n_params/1e6:.1f}M parameters  |  config: {config}")

    if ddp:
        model = DDP(model, device_ids=[local_rank])

    # ---- Optimizer --------------------------------------------------------
    # Separate weight-decay and no-decay groups
    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() < 2 or "norm" in name or "bias" in name:
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    optimizer = torch.optim.adamw.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
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
            print(f"[Resume] Loaded checkpoint from {args.resume_from}, step={start_step}")

    # ---- Dataset / Dataloader ---------------------------------------------
    dataset = TextDataset(args.dataset_path, tokenizer, args.seq_len)

    if ddp:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # ---- TensorBoard ------------------------------------------------------
    writer = None
    if is_master:
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tb"))

    # ---- Training loop ----------------------------------------------------
    model.train()
    data_iter = iter(dataloader)
    epoch = 0

    last_ckpt_time = time.time()
    global_step = start_step
    accum_loss = 0.0

    if is_master:
        print(f"[Train] Starting from step {global_step}, max_steps={args.max_steps}")
        print(f"[Train] Effective batch size = {args.batch_size * args.grad_accum_steps * world_size}")

    while global_step < args.max_steps:
        # Grab a micro-batch, cycling over epochs
        try:
            x, y = next(data_iter)
        except StopIteration:
            epoch += 1
            if ddp:
                assert sampler is not None
                sampler.set_epoch(epoch)
            data_iter = iter(dataloader)
            x, y = next(data_iter)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Forward / backward (gradient accumulation)
        if ddp and ((global_step + 1) % args.grad_accum_steps != 0):
            sync_context = model.no_sync()
        else:
            sync_context = contextlib.nullcontext()

        with sync_context:
            with autocast("cuda", enabled=args.use_amp, dtype=torch.float16):
                _, loss = model(
                    input_ids=x,
                    attention_mask=None,
                    start_pos=0,
                    cache_k=None,
                    cache_v=None,
                    labels=y
                )
                loss = loss / args.grad_accum_steps
            scaler.scale(loss).backward()
            accum_loss += loss.item()

        # Optimizer step after accumulation
        if (global_step + 1) % args.grad_accum_steps == 0:
            # LR schedule
            lr = get_lr(global_step // args.grad_accum_steps, args.warmup_steps, args.max_steps // args.grad_accum_steps, args.max_lr, args.min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # Logging
            if is_master and (global_step + 1) % (args.log_every * args.grad_accum_steps) == 0:
                avg_loss = accum_loss / args.log_every  # loss was already divided by accum
                # But we accumulated `log_every` optimizer steps worth
                # Recompute properly:
                opt_step = (global_step + 1) // args.grad_accum_steps
                print(f"step={opt_step}  loss={accum_loss:.4f}  lr={lr:.2e}")
                if is_master and writer is not None and (global_step + 1) % (args.log_every * args.grad_accum_steps) == 0:
                    writer.add_scalar("train/loss", accum_loss, opt_step)
                    writer.add_scalar("train/lr", lr, opt_step)
                accum_loss = 0.0

        global_step += 1

        # ---- Checkpointing by steps --------------------------------------
        if is_master and args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
            ckpt_path = os.path.join(args.output_dir, f"ckpt_step{global_step}.pt")
            save_checkpoint(model, optimizer, global_step, config, ckpt_path)
            print(f"[Checkpoint] Saved {ckpt_path}")

        # ---- Checkpointing by wall-clock time -----------------------------
        if is_master and args.save_every_seconds > 0:
            now = time.time()
            if now - last_ckpt_time >= args.save_every_seconds:
                ckpt_path = os.path.join(args.output_dir, f"ckpt_time_{int(now)}.pt")
                save_checkpoint(model, optimizer, global_step, config, ckpt_path)
                print(f"[Checkpoint] Timed save {ckpt_path}")
                last_ckpt_time = now

    # ---- Final save -------------------------------------------------------
    if is_master:
        final_path = os.path.join(args.output_dir, "ckpt_final.pt")
        save_checkpoint(model, optimizer, global_step, config, final_path)
        print(f"[Done] Final checkpoint saved to {final_path}")
        if is_master and writer is not None:
            writer.close()

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()