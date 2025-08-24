#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Count MHQA datasets (Hugging Face Datasets) in the exact order of the original paper's Table 6.
- No extra libraries required beyond `datasets`.
- Prints a compact, easy-to-read table and verifies counts against expected values from the paper.

Usage:
    uv run python get_mhqa_datasets.py

Notes:
- We intentionally load specific configs/splits that correspond to the evaluation sets used in Table 6.
- If any dataset fails to load, we capture the error and continue.
"""

import argparse
from datasets import load_dataset

# Table 6 order (exact): NQ, TriviaQA, PopQA, HotpotQA, 2Wiki, MuSiQue, Bamboogle
# Map each display row to (hf_id, config, split, expected_count)
TABLE6_DATASETS = [
    ("NQ (nq_open)",              "nq_open",                     None,        "validation", 3610),
    ("TriviaQA (rc)",             "trivia_qa",                   "rc",       "validation", 17944),
    ("PopQA",                      "akariasai/PopQA",            None,        "test",       14267),
    ("HotpotQA (distractor)",     "hotpot_qa",                   "distractor","validation", 7405),
    ("2Wiki",                      "cmriat/2wikimultihopqa",     None,        "validation", 12576),
    ("MuSiQue",                   "fladhak/musique",            None,        "validation", 2417),
    ("Bamboogle",                 "chiayewken/bamboogle",       None,        "test",       125),
]


def count_split(hf_id: str, split: str, config: str | None = None) -> int:
    ds = load_dataset(hf_id, config, split=split) if config else load_dataset(hf_id, split=split)
    return len(ds)


def main():
    parser = argparse.ArgumentParser(description="Count MHQA datasets in Table 6 order")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-dataset loading logs")
    args = parser.parse_args()

    # Header
    print("Dataset                         HF ID                           Config      Split         Count   Check")
    print("-" * 98)

    all_ok = True

    for disp_name, hf_id, config, split, expected in TABLE6_DATASETS:
        try:
            if not args.quiet:
                print(f"Loading dataset: {hf_id}{'/' + config if config else ''}, split={split}")
            cnt = count_split(hf_id, split, config)
            status = "✓" if (expected is None or cnt == expected) else f"× (expected {expected})"
            if expected is not None and cnt != expected:
                all_ok = False
            print(f"{disp_name:<30} {hf_id:<30} {str(config or '-'):>10}   {split:<12} {cnt:<7} {status}")
        except Exception as e:
            all_ok = False
            print(f"{disp_name:<30} {hf_id:<30} {str(config or '-'):>10}   {split:<12} {'-':<7} Failed: {e}")

    print("\nSummary:")
    print("  ✔ All counts match Table 6" if all_ok else "  ✗ Mismatch or load failure detected. See rows above.")


if __name__ == "__main__":
    main()
