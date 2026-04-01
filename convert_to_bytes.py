#!/usr/bin/env python3
"""Convert sp1024 BPE shards to raw UTF-8 byte shards for H-net training.

Usage (on your RunPod H100):
    # First download sp1024 data if not already done:
    python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

    # Then convert:
    python3 convert_to_bytes.py

This reads from ./data/datasets/fineweb10B_sp1024/
and writes to   ./data/datasets/fineweb10B_bytes/
"""
import glob
import os
import struct
import sys
from pathlib import Path

import numpy as np
import sentencepiece as spm

MAGIC = 20240520
VERSION = 1
HEADER_INTS = 256

SRC_DIR = Path("./data/datasets/fineweb10B_sp1024")
DST_DIR = Path("./data/datasets/fineweb10B_bytes")
TOKENIZER = Path("./data/tokenizers/fineweb_1024_bpe.model")


def read_shard(path: Path) -> np.ndarray:
    """Read a shard and return uint16 token IDs."""
    header = np.fromfile(path, dtype="<i4", count=HEADER_INTS)
    assert header[0] == MAGIC, f"Bad magic in {path}"
    assert header[1] == VERSION, f"Bad version in {path}"
    num_tokens = int(header[2])
    header_bytes = HEADER_INTS * 4
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    assert tokens.size == num_tokens, f"Short read: {path}"
    return tokens


def write_shard(path: Path, byte_values: np.ndarray):
    """Write a shard with byte values (0-255) stored as uint16."""
    header = np.zeros(HEADER_INTS, dtype="<i4")
    header[0] = MAGIC
    header[1] = VERSION
    header[2] = len(byte_values)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(byte_values.astype("<u2").tobytes())


def convert_shard(sp: spm.SentencePieceProcessor, src: Path, dst: Path):
    """Decode BPE tokens -> text -> UTF-8 bytes, write new shard."""
    tokens = read_shard(src)
    token_ids = tokens.tolist()

    # Decode BPE tokens back to text
    # SentencePiece decode handles BOS (id=1) gracefully
    text = sp.decode(token_ids)

    # Encode text as raw UTF-8 bytes
    raw_bytes = text.encode("utf-8", errors="replace")
    byte_array = np.frombuffer(raw_bytes, dtype=np.uint8)

    write_shard(dst, byte_array)
    return len(token_ids), len(byte_array)


def main():
    if not TOKENIZER.exists():
        print(f"ERROR: Tokenizer not found at {TOKENIZER}")
        print("Run: python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10")
        sys.exit(1)

    if not SRC_DIR.exists():
        print(f"ERROR: Source data not found at {SRC_DIR}")
        print("Run: python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10")
        sys.exit(1)

    DST_DIR.mkdir(parents=True, exist_ok=True)

    sp = spm.SentencePieceProcessor(str(TOKENIZER))
    print(f"Loaded tokenizer: vocab_size={sp.get_piece_size()}")

    # Find all shards (train + val)
    shards = sorted(glob.glob(str(SRC_DIR / "fineweb_*.bin")))
    if not shards:
        print(f"ERROR: No shard files found in {SRC_DIR}")
        sys.exit(1)

    print(f"Found {len(shards)} shards to convert")
    print(f"Source: {SRC_DIR}")
    print(f"Destination: {DST_DIR}")
    print()

    total_bpe_tokens = 0
    total_bytes = 0

    for i, src_path in enumerate(shards):
        src = Path(src_path)
        dst = DST_DIR / src.name  # same filename, different directory

        n_tokens, n_bytes = convert_shard(sp, src, dst)
        total_bpe_tokens += n_tokens
        total_bytes += n_bytes

        ratio = n_bytes / n_tokens if n_tokens > 0 else 0
        print(f"[{i+1}/{len(shards)}] {src.name}: {n_tokens:,} BPE tokens -> {n_bytes:,} bytes (ratio {ratio:.2f}x)")

    print()
    print(f"Done! Converted {len(shards)} shards")
    print(f"  Total BPE tokens: {total_bpe_tokens:,}")
    print(f"  Total bytes:      {total_bytes:,}")
    print(f"  Avg bytes/token:  {total_bytes/total_bpe_tokens:.2f}")
    print()
    print(f"To train with H-net, run:")
    print(f"  DATA_PATH=./data/datasets/fineweb10B_bytes VOCAB_SIZE=256 \\")
    print(f"  ITERATIONS=500 VAL_LOSS_EVERY=100 TRAIN_LOG_EVERY=10 \\")
    print(f"  SPACE_CURRICULUM_STEPS=200 MAX_WALLCLOCK_SECONDS=300 \\")
    print(f"  torchrun --standalone --nproc_per_node=1 train_gpt.py")


if __name__ == "__main__":
    main()
