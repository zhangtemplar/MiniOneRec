#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict

"""
Convert JSON item-to-semantic-ID mapping to Parquet format for rankagi pipeline.

This script converts JSON mapping files (with hierarchical semantic tokens like
<a_203>, <b_37>, etc.) into the Parquet format expected by the rankagi data pipeline.

Input JSON format:
{
  "54513855": ["<a_203>", "<b_37>", "<c_20>", "<d_250>", "<e_121>", "<f54>"],
  ...
}

Output Parquet schema:
- item_idx (Int64): Item ID
- semantic_id (String): Space-separated REC tokens (e.g., "REC203 REC37 REC20 REC250 REC121 REC54")
- semantic_id_list (List[Int64]): Array of token IDs (e.g., [203, 37, 20, 250, 121, 54])

COLLISION HANDLING:
When merging train and eval JSON files:
- If the same item_idx appears in both files, the TRAIN version is kept
- The script reports collision count and rate
- Final output contains unique items only (no duplicates)

USAGE:
------
# Train only:
python -m data.convert_json_mapping_to_parquet \
    --train_json ~/local/data/my_mappings/train_mapping.json \
    --output ~/local/data/merrec/cpt/v2/merrec_item_to_sid.parquet \
    --validate

# Train + Eval (merged):
python -m data.convert_json_mapping_to_parquet \
    --train_json ~/local/data/my_mappings/train_mapping.json \
    --eval_json ~/local/data/my_mappings/eval_mapping.json \
    --output ~/local/data/merrec/cpt/v2/merrec_item_to_sid.parquet \
    --validate

Requirements:
  pip install polars pyarrow
"""
import logging
import argparse
import ijson
import re
from pathlib import Path
from typing import Iterator, Optional, Set, Tuple
import polars as pl
import gc


logger: logging.Logger = logging.getLogger(__name__)


def parse_semantic_token(token: str) -> Optional[int]:
    """Parse semantic token to extract numeric ID."""
    token = token.strip("<>")

    # Standard: "a_203"
    match = re.match(r"^[a-h]_(\d+)$", token)
    if match:
        return int(match.group(1))

    # Typo: "f54"
    match = re.match(r"^[a-h](\d+)$", token)
    if match:
        return int(match.group(1))

    return None


def convert_semantic_tokens(token_list):
    """Convert list of tokens to rankagi format."""
    token_ids = []

    for token in token_list:
        token_id = parse_semantic_token(token)
        if token_id is None:
            return None, None
        token_ids.append(token_id)

    semantic_id_list = token_ids
    semantic_id = " ".join(f"REC{tid}" for tid in token_ids)

    return semantic_id_list, semantic_id


def stream_json_items(json_path: str) -> Iterator[Tuple[int, list]]:
    """
    Stream JSON items using ijson (TRUE streaming, constant memory).

    Parses: {"item1": [...], "item2": [...], ...}
    Yields: (item_idx, token_list) one at a time

    Memory: ~Constant per item (~1KB), regardless of total file size!
    """
    path = Path(json_path).expanduser()

    logger.error(f"  Streaming from: {path} (using ijson)")

    count = 0
    with open(path, 'rb') as f:
        # ijson.kvitems iterates over key-value pairs in a JSON object
        # This does NOT load the entire dict into memory!
        parser = ijson.kvitems(f, '')

        for item_id_str, token_list in parser:
            try:
                item_idx = int(item_id_str)
                yield item_idx, token_list

                count += 1
                if count % 1000000 == 0:
                    logger.error(f"    Streamed {count:,} items...")

            except (ValueError, TypeError):
                continue


def extract_item_ids_streaming(json_path: str) -> Set[int]:
    """
    Extract item IDs using TRUE streaming (constant memory).

    Memory: Only stores the set of IDs (~10 bytes per ID = 500MB for 50M items)
    """
    logger.error(f"\nExtracting item IDs from: {json_path}")

    item_ids = set()

    for item_idx, _ in stream_json_items(json_path):
        item_ids.add(item_idx)

    logger.error(f"  Extracted {len(item_ids):,} unique item IDs")

    import sys
    mem_mb = sys.getsizeof(item_ids) / (1024 * 1024)
    logger.error(f"  Memory usage: ~{mem_mb:.1f} MB")

    return item_ids


def process_json_streaming(
    json_path: str,
    output_path: Path,
    chunk_size: int,
    split_name: str,
    exclude_ids: Optional[Set[int]] = None,
) -> Tuple[int, int, int]:
    """Process JSON using TRUE streaming (constant memory)."""

    logger.error(f"\nProcessing {split_name} data...")
    logger.error(f"  Input: {json_path}")
    logger.error(f"  Chunk size: {chunk_size:,}")

    chunk_records = []
    chunk_files = []
    success_count = 0
    fail_count = 0
    collision_count = 0
    chunk_idx = 0

    exclude_ids = exclude_ids or set()

    for item_idx, token_list in stream_json_items(json_path):
        # Check collision
        if item_idx in exclude_ids:
            collision_count += 1
            continue

        # Convert tokens
        semantic_id_list, semantic_id = convert_semantic_tokens(token_list)

        if semantic_id_list is None:
            fail_count += 1
            continue

        chunk_records.append({
            "item_idx": item_idx,
            "semantic_id": semantic_id,
            "semantic_id_list": semantic_id_list,
        })
        success_count += 1

        # Write chunk when full
        if len(chunk_records) >= chunk_size:
            chunk_file = output_path.parent / f"{output_path.stem}_chunk_{chunk_idx:06d}.parquet"
            df_chunk = pl.DataFrame(chunk_records)
            df_chunk.write_parquet(chunk_file)
            chunk_files.append(chunk_file)

            if chunk_idx % 10 == 0 or chunk_idx < 3:
                logger.error(f"  Chunk {chunk_idx}: {len(chunk_records):,} records → {chunk_file.name}")

            chunk_records = []
            chunk_idx += 1
            gc.collect()

    # Write final chunk
    if chunk_records:
        chunk_file = output_path.parent / f"{output_path.stem}_chunk_{chunk_idx:06d}.parquet"
        df_chunk = pl.DataFrame(chunk_records)
        df_chunk.write_parquet(chunk_file)
        chunk_files.append(chunk_file)
        logger.error(f"  Chunk {chunk_idx}: {len(chunk_records):,} records → {chunk_file.name}")

    logger.error(f"\n  {split_name} summary:")
    logger.error(f"    Converted: {success_count:,}")
    logger.error(f"    Failed: {fail_count:,}")
    if collision_count > 0:
        logger.error(f"    Collisions: {collision_count:,}")
    logger.error(f"    Chunks: {len(chunk_files)}")

    # Concatenate chunks using LAZY evaluation (avoids loading all into memory)
    if len(chunk_files) > 0:
        logger.error(f"\n  Concatenating {len(chunk_files)} chunks (using lazy evaluation)...")

        # Use glob pattern for lazy reading
        chunk_pattern = str(output_path.parent / f"{output_path.stem}_chunk_*.parquet")

        # Lazy scan all chunks, sort, and write in one pass (streaming)
        # This NEVER loads all chunks into memory at once!
        df_lazy = pl.scan_parquet(chunk_pattern)
        df_lazy = df_lazy.sort("item_idx")

        # sink_parquet writes directly without collecting into memory
        df_lazy.sink_parquet(output_path)

        # Delete chunk files
        for chunk_file in chunk_files:
            chunk_file.unlink()

        # Count items (lazy)
        item_count = pl.scan_parquet(output_path).select(pl.len()).collect().item()
        logger.error(f"  ✓ Final file: {output_path} ({item_count:,} items)")

        del df_lazy
        gc.collect()

    return success_count, fail_count, collision_count


def validate_parquet(parquet_path: Path) -> None:
    """Validate output (lazy evaluation for large files)."""
    logger.error("\n" + "=" * 80)
    logger.error("Validation Report")
    logger.error("=" * 80)

    df = pl.scan_parquet(parquet_path)

    stats = df.select([
        pl.len().alias("total"),
        pl.col("item_idx").n_unique().alias("unique"),
        pl.col("item_idx").is_null().sum().alias("nulls"),
    ]).collect()

    logger.error(f"✓ Total items: {stats['total'][0]:,}")
    logger.error(f"✓ Unique items: {stats['unique'][0]:,}")
    logger.error(f"✓ Null items: {stats['nulls'][0]:,}")

    if stats['total'][0] != stats['unique'][0]:
        logger.error(f"⚠ Warning: {stats['total'][0] - stats['unique'][0]:,} duplicates")

    # Sample
    samples = df.head(3).collect()
    logger.error("\nSample records:")
    for row in samples.iter_rows(named=True):
        logger.error(f"  {row['item_idx']}: {row['semantic_id']}")


def main():
    parser = argparse.ArgumentParser(description="Convert JSON to Parquet (TRUE streaming with ijson)")
    parser.add_argument("--train_json", required=True, help="Train JSON file")
    parser.add_argument("--eval_json", default=None, help="Eval JSON file (optional)")
    parser.add_argument("--output", required=True, help="Output parquet")
    parser.add_argument("--chunk_size", type=int, default=100000, help="Chunk size")
    parser.add_argument("--validate", action="store_true", help="Validate output")

    args = parser.parse_args()

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.error("=" * 80)
    logger.error("JSON → Parquet Conversion (TRUE STREAMING with ijson)")
    logger.error("=" * 80)
    logger.error("Memory: ~1-2GB constant (does NOT load entire JSON!)")
    logger.error(f"Output: {output_path}")
    logger.error("")

    # Extract train IDs (streaming)
    train_ids = extract_item_ids_streaming(args.train_json)

    # Process train (streaming)
    train_path = output_path.parent / f"{output_path.stem}_train_temp.parquet"
    train_success, train_fail, _ = process_json_streaming(
        args.train_json, train_path, args.chunk_size, "train", None
    )

    total_success = train_success
    total_fail = train_fail

    # Process eval (streaming, skip collisions)
    if args.eval_json:
        eval_path = output_path.parent / f"{output_path.stem}_eval_temp.parquet"
        eval_success, eval_fail, collisions = process_json_streaming(
            args.eval_json, eval_path, args.chunk_size, "eval", train_ids
        )
        total_success += eval_success
        total_fail += eval_fail

        # Merge
        logger.error("\n" + "=" * 80)
        logger.error("Merging Train + Eval")
        logger.error("=" * 80)
        logger.error(f"Train: {train_success:,}")
        logger.error(f"Eval: {eval_success:,}")
        logger.error(f"Collisions (kept train): {collisions:,}")
        logger.error(f"Unique after merge: {train_success + eval_success:,}")

        if collisions > 0 and (eval_success + collisions) > 0:
            pct = collisions / (eval_success + collisions) * 100
            logger.error(f"Collision rate: {pct:.2f}%")

        # Merge using LAZY evaluation (never loads full datasets into memory)
        logger.error("Merging parquet files (lazy evaluation)...")

        # Lazy scan both files, concatenate, sort, and write in one streaming pass
        df_lazy = pl.concat([
            pl.scan_parquet(train_path),
            pl.scan_parquet(eval_path),
        ])
        df_lazy = df_lazy.sort("item_idx")

        # sink_parquet writes directly without collecting into memory
        df_lazy.sink_parquet(output_path)

        # Delete temp files
        train_path.unlink()
        eval_path.unlink()

        # Count items (lazy)
        item_count = pl.scan_parquet(output_path).select(pl.len()).collect().item()
        logger.error(f"✓ Merged: {item_count:,} items")

        del df_lazy
        gc.collect()
    else:
        train_path.rename(output_path)

    # Summary
    logger.error("\n" + "=" * 80)
    logger.error("✅ Conversion Complete!")
    logger.error("=" * 80)
    logger.error(f"Converted: {total_success:,}")
    logger.error(f"Failed: {total_fail:,}")
    logger.error(f"Output: {output_path}")
    logger.error(f"Size: {output_path.stat().st_size / (1024**2):.2f} MB")

    if args.validate:
        validate_parquet(output_path)

    logger.error("\nNext: Use this file with data/item_text.py")


if __name__ == "__main__":
    main()