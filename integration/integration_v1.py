import os
import json
import argparse
from typing import List, Literal, Dict, Any

from datasets import load_dataset
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

from pathlib import Path

from multi_agent.multi_agent_detect_critic_edit import load_indices, Config, run_one, to_submission_line
#from 


def run_integration()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation", "test"])
    parser.add_argument("--limit", type=int, default=0, help="0 means all")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--detector_model", type=str, default="gpt-5.2")
    parser.add_argument("--critic_model", type=str, default="gpt-5.2")
    parser.add_argument("--editor_model", type=str, default="gpt-5.2")
    parser.add_argument("--n_best", type=int, default=3)
    parser.add_argument("--simple_indices", type=str, default="baseline-experiment/sampled_test_indices.json")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Please set OPENAI_API_KEY env var.")

    client = OpenAI()

    ds = load_dataset("mkieffer/MEDEC", split=args.split)
    # if args.limit and args.limit > 0:
    #     ds = ds.select(range(min(args.limit, len(ds))))

    loaded_indices = load_indices(args.simple_indices)
    ds = ds.select(loaded_indices)
    num_data = len(ds)
    
    cfg = Config(
        detector_model=args.detector_model,
        critic_model=args.critic_model,
        editor_model=args.editor_model,
        n_best=args.n_best,
    )

    if args.out is None:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = f"medec_multi-agent_submission_integrated_{args.split}_{ts}.txt"

    with open(args.out, "w", encoding="utf-8") as f:
        for row in tqdm(ds, desc=f"MEDEC {args.split} -> integrated submission"):
            try:
                line = run_one(client, row, cfg)
            except Exception:
                # strict fail-safe: keep format valid
                line = to_submission_line(row.get("text_id", "UNKNOWN"), 0, -1, "NA")
            f.write(line + "\n")

    print(f"[OK] Saved: {args.out}")

if __name__ == "__main__":
    print("a")