#!/usr/bin/env python3
"""
MEDIQA-CORR 2024 — Ablation Study Runner

Tests all meaningful component combinations to isolate the contribution of
each approach (RAG, ICL, Multi-Agent) and their interactions.

Ablation matrix (7 combinations):
  ┌─────┬─────┬────┬──────────────────────────────────────────┐
  │ RAG │ ICL │ MA │ Label                                    │
  ├─────┼─────┼────┼──────────────────────────────────────────┤
  │  ✓  │     │    │ rag                  (from prior run)    │
  │     │  ✓  │    │ icl_dynamic          (from prior run)    │
  │     │     │ ✓  │ multi_agent          (from prior run)    │
  │  ✓  │  ✓  │    │ rag+icl   (single-pass, no MA)          │
  │  ✓  │     │ ✓  │ rag+ma    (MA with drug context only)   │
  │     │  ✓  │ ✓  │ icl+ma    (MA with examples only)       │
  │  ✓  │  ✓  │ ✓  │ combined  (from prior run)              │
  └─────┴─────┴────┴──────────────────────────────────────────┘

The first three and last one are taken from existing submission files (no API
calls needed). Only the three middle rows require new LLM calls.

Also computes BLEURT for all methods (including previously run ones).

Usage:
  python integration/run_ablation.py --model gpt-4.1
  python integration/run_ablation.py --model gpt-4.1 --num-samples 100
  python integration/run_ablation.py --model gpt-4.1 --full
  python integration/run_ablation.py --existing-only   # just re-evaluate prior outputs with BLEURT
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Script lives in integration/, so parent.parent is the repo root
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

INDICES_FILE   = REPO_ROOT / "baseline-experiment" / "sampled_test_indices.json"
ABLATION_DIR   = Path(__file__).parent / "ablation"

# Previously computed submission files (reused for the 50-sample subset only)
PRIOR_OUTPUTS = Path(__file__).parent / "outputs" / "20260314_225219"
PRIOR_COMBINED = Path(__file__).parent / "outputs" / "20260314_232020"


# ── Utilities (shared with run_integrated.py) ─────────────────────────────────

def _load_module_from_file(name: str, filepath: Path):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def load_icl_module():
    return _load_module_from_file(
        "in_context_learning",
        REPO_ROOT / "in-context-learning" / "in-context-learning.py",
    )

def load_multiagent_module():
    return _load_module_from_file(
        "multi_agent_dce",
        REPO_ROOT / "multi-agent" / "multi-agent-detect-critic-edit.py",
    )

def load_evaluate_module():
    return _load_module_from_file(
        "evaluate",
        REPO_ROOT / "evaluation" / "evaluate.py",
    )

def _bootstrap_openai_key() -> None:
    if os.environ.get("OPENAI_API_KEY"):
        return
    key_path = Path.home() / "env" / "openai_secret_key.txt"
    if key_path.exists():
        key = key_path.read_text(encoding="utf-8").strip()
        if key:
            os.environ["OPENAI_API_KEY"] = key

def load_sampled_indices() -> List[int]:
    return json.loads(INDICES_FILE.read_text(encoding="utf-8"))["indices"]

def load_hf_dataset(dataset_name: str, split: str, indices: Optional[List[int]],
                    num_samples: Optional[int] = None):
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split=split)
    if indices is not None:
        ds = ds.select(indices)
    if num_samples is not None:
        ds = ds.select(range(min(num_samples, len(ds))))
    return ds


# ── New ablation runs ─────────────────────────────────────────────────────────

def run_ablation_variant(
    dataset,
    model: str,
    use_rag: bool,
    use_icl: bool,
    use_multiagent: bool,
    k_shot: int,
    output_path: Path,
    icl_train_dataset: str = "mkieffer/MEDEC",
    icl_train_split: str = "train",
    icl_train_dataset_config: str = "default",
    # shared state to avoid re-loading heavy objects
    _shared: Optional[Dict] = None,
) -> Path:
    """
    Run one ablation variant using CombinedPredictor with the given component flags.
    Pass _shared dict to cache train_items/embeddings across multiple variants.
    """
    from pipeline.combined import CombinedPredictor

    predictor = CombinedPredictor(
        model=model,
        k_shot=k_shot,
        use_rag=use_rag,
        use_icl=use_icl,
        use_multiagent=use_multiagent,
    )

    # Load and embed training data (cached via _shared to avoid re-embedding)
    if use_icl:
        if _shared is not None and "train_items" in _shared:
            predictor._train_items      = _shared["train_items"]
            predictor._train_embeddings = _shared["train_embeddings"]
            print("  (reusing cached train embeddings)")
        else:
            icl = load_icl_module()
            train_items = icl.load_hf_dataset_split(
                icl_train_dataset, icl_train_dataset_config, icl_train_split
            )
            predictor.load_train_data(train_items)
            if _shared is not None:
                _shared["train_items"]      = predictor._train_items
                _shared["train_embeddings"] = predictor._train_embeddings

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(dataset)
    label = predictor.label
    with output_path.open("w", encoding="utf-8") as fh:
        for i, row in enumerate(dataset):
            try:
                line = predictor.predict(dict(row))
            except Exception as exc:
                print(f"    WARN [{label}] {row.get('text_id','?')} — {exc}")
                line = f"{row.get('text_id','UNKNOWN')} 0 -1 NA"
            fh.write(line + "\n")
            print(f"  [{label}] {i+1}/{n}  {line[:80]}")

    print(f"  → written: {output_path}")
    return output_path


# ── Evaluation (with BLEURT) ──────────────────────────────────────────────────

def evaluate_submission(
    submission_path: Path,
    ref_dataset,
    include_bleurt: bool = True,
    include_bertscore: bool = True,
) -> Dict[str, float]:
    ev = load_evaluate_module()
    ref_corrections, ref_flags, ref_sent_id = ev.parse_reference_dataset(ref_dataset)
    cand_corrections, cand_flags, cand_sent_id = ev.parse_run_submission_file(
        str(submission_path)
    )
    accuracy   = ev.compute_accuracy(ref_flags, ref_sent_id, cand_flags, cand_sent_id)
    refs, preds, counters = ev.get_nlg_eval_data(ref_corrections, cand_corrections)
    nlg = ev.compute_nlg_metrics(
        refs, preds, counters,
        include_bleurt=include_bleurt,
        include_bertscore=include_bertscore,
    )
    return {**accuracy, **nlg}


# ── Report ────────────────────────────────────────────────────────────────────

def print_and_save_report(
    results: Dict[str, Dict[str, float]],
    out_path: Path,
    model: str,
    n_samples: int,
    dataset: str,
) -> None:
    method_labels = {
        "baseline":   "Baseline (zero-shot)",
        "rag":        "RAG only",
        "icl_dynamic":"ICL-Dynamic k=5",
        "multi_agent":"Multi-Agent only",
        "rag+icl":    "RAG + ICL (no MA)",
        "rag+ma":     "RAG + MA  (no ICL)",
        "icl+ma":     "ICL + MA  (no RAG)",
        "combined":   "RAG + ICL + MA  ←FULL",
    }
    metrics = [
        ("Flag Acc",         "Error Flags Accuracy"),
        ("Sentence Acc",     "Error Sentence Detection Accuracy"),
        ("ROUGE-1",          "ROUGE1"),
        ("ROUGE-2",          "ROUGE2"),
        ("ROUGE-L",          "ROUGEL"),
        ("BERTScore",        "BERTSCORE"),
        ("BLEURT",           "BLEURT"),
        ("ROUGE1 Comp",      "ROUGE1_Composite"),
        ("BERTScr Comp",     "BERTSCORE_Composite"),
        ("BLEURT Comp",      "BLEURT_Composite"),
        ("Agg Score",        "AggregateScore"),
        ("Agg Composite",    "AggregateComposite"),
    ]

    ordered = ["baseline", "rag", "icl_dynamic", "multi_agent", "rag+icl", "rag+ma", "icl+ma", "combined"]
    present = [m for m in ordered if m in results]

    lines = []
    lines.append("MEDIQA-CORR 2024 — Ablation Study Results")
    lines.append("=" * 100)
    lines.append(f"Model   : {model}")
    lines.append(f"Dataset : {dataset}  ({n_samples} samples)")
    lines.append(f"Date    : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("Component matrix:")
    lines.append("  Label             RAG   ICL   MA")
    lines.append("  ─────────────────────────────────")
    combos = {
        "baseline":   (False, False, False),
        "rag":        (True,  False, False),
        "icl_dynamic":(False, True,  False),
        "multi_agent":(False, False, True),
        "rag+icl":    (True,  True,  False),
        "rag+ma":     (True,  False, True),
        "icl+ma":     (False, True,  True),
        "combined":   (True,  True,  True),
    }
    for m in present:
        r, i, a = combos.get(m, (False, False, False))
        lines.append(
            f"  {method_labels.get(m, m):<20} {'✓' if r else '·':^5} {'✓' if i else '·':^5} {'✓' if a else '·':^5}"
        )
    lines.append("")

    col0 = 18
    cw   = 14
    labels_row = [method_labels.get(m, m) for m in present]
    header = f"{'Metric':<{col0}}" + "".join(f"{l:>{cw}}" for l in labels_row)
    sep    = "-" * len(header)
    lines.append(header)
    lines.append(sep)

    for mlabel, key in metrics:
        row = f"{mlabel:<{col0}}"
        for m in present:
            val = results[m].get(key)
            row += "           N/A" if val is None else f"{val:>{cw}.4f}"
        lines.append(row)

    lines.append("")
    lines.append("ABLATION ANALYSIS (delta vs. Multi-Agent baseline):")
    if "multi_agent" in results:
        base = results["multi_agent"]
        for m in present:
            if m == "multi_agent":
                continue
            r = results[m]
            delta_flag = r.get("Error Flags Accuracy", 0) - base.get("Error Flags Accuracy", 0)
            delta_sent = r.get("Error Sentence Detection Accuracy", 0) - base.get("Error Sentence Detection Accuracy", 0)
            delta_agg  = r.get("AggregateScore", 0) - base.get("AggregateScore", 0)
            delta_comp = r.get("AggregateComposite", 0) - base.get("AggregateComposite", 0)
            lines.append(
                f"  {method_labels.get(m, m):<28}  "
                f"FlagAcc {delta_flag:+.4f}  "
                f"SentAcc {delta_sent:+.4f}  "
                f"AggScore {delta_agg:+.4f}  "
                f"AggComp {delta_comp:+.4f}"
            )

    report = "\n".join(lines) + "\n"
    print(report)
    out_path.write_text(report, encoding="utf-8")
    print(f"Report saved to: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="MEDIQA-CORR Ablation Study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--icl-k", type=int, default=5)
    parser.add_argument("--dataset", default="mkieffer/MEDEC-MS")
    parser.add_argument("--split",   default="test")
    parser.add_argument("--full",    action="store_true",
                        help="Use full test set (all 597 samples) — re-runs all 7 variants fresh")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Override sample count (e.g. --num-samples 100)")
    parser.add_argument("--no-bleurt",     action="store_true")
    parser.add_argument("--no-bertscore",  action="store_true")
    parser.add_argument("--existing-only", action="store_true",
                        help="Skip new LLM runs; only (re-)evaluate prior outputs with BLEURT")
    parser.add_argument("--output-dir", type=Path, default=ABLATION_DIR)
    args = parser.parse_args()

    _bootstrap_openai_key()

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load indices / dataset
    indices = None if args.full else load_sampled_indices()
    print(f"\nLoading test dataset: {args.dataset} / {args.split}")
    ref_dataset = load_hf_dataset(args.dataset, args.split, indices, args.num_samples)
    n_samples   = len(ref_dataset)
    print(f"  {n_samples} samples")

    # ── Locate prior submission files ─────────────────────────────────────────
    # Prior files come from 50-sample runs.  When running the full dataset (or a
    # custom sample count) we must re-run everything from scratch — reusing
    # 50-sample outputs against a larger reference set would give wrong metrics.
    use_prior = not (args.full or args.num_samples is not None)

    prior_files = {
        "rag":         PRIOR_OUTPUTS / "rag_pubmedbert_gpt-4.1.txt",
        "icl_dynamic": sorted((PRIOR_OUTPUTS / "icl").glob("*dynamic*.txt"))[-1]
                       if list((PRIOR_OUTPUTS / "icl").glob("*dynamic*.txt")) else None,
        "multi_agent": PRIOR_OUTPUTS / "multi_agent_gpt-4.1.txt",
        "combined":    PRIOR_COMBINED / "combined_rag_icl_k5_ma_gpt-4.1.txt",
    } if use_prior else {}

    submission_files: Dict[str, Optional[Path]] = {}

    # Copy valid prior files into this run's directory for clean bookkeeping
    import shutil
    for key, src in prior_files.items():
        if src and src.exists():
            dst = run_dir / f"{key}.txt"
            shutil.copy2(src, dst)
            submission_files[key] = dst
            print(f"  Reusing prior output for [{key}]: {src.name}")
        else:
            print(f"  WARNING: prior output for [{key}] not found at {src}")
            submission_files[key] = None

    # ── New ablation runs ─────────────────────────────────────────────────────
    # When running the full dataset, run all 7 variants fresh.
    # When reusing prior files, only run the 3 pairwise variants not previously computed.
    if args.existing_only:
        new_variants = []
    elif not use_prior:
        # Full / custom-size run — generate all 8 from scratch
        new_variants = [
            ("baseline",    False, False, False),
            ("rag",         True,  False, False),
            ("icl_dynamic", False, True,  False),
            ("multi_agent", False, False, True),
            ("rag+icl",     True,  True,  False),
            ("rag+ma",      True,  False, True),
            ("icl+ma",      False, True,  True),
            ("combined",    True,  True,  True),
        ]
    else:
        new_variants = [
            # (label,  use_rag, use_icl, use_ma)
            ("rag+icl", True,  True,  False),
            ("rag+ma",  True,  False, True),
            ("icl+ma",  False, True,  True),
        ]

    shared: Dict[str, Any] = {}  # cache train embeddings across variants

    for label, use_rag, use_icl, use_ma in new_variants:
        print(f"\n{'='*60}")
        print(f"Running: {label}  (RAG={use_rag} ICL={use_icl} MA={use_ma})")
        print("=" * 60)
        out = run_dir / f"{label}.txt"
        try:
            submission_files[label] = run_ablation_variant(
                dataset=ref_dataset,
                model=args.model,
                use_rag=use_rag,
                use_icl=use_icl,
                use_multiagent=use_ma,
                k_shot=args.icl_k,
                output_path=out,
                _shared=shared,
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            submission_files[label] = None

    # ── Evaluation ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Evaluating all methods" + (" (with BLEURT)" if not args.no_bleurt else ""))
    print("=" * 60)

    results: Dict[str, Dict[str, float]] = {}
    for name, path in submission_files.items():
        if path is None or not path.exists():
            print(f"\n  Skipping [{name}]: no output file")
            continue
        print(f"\n  [{name}]")
        try:
            metrics = evaluate_submission(
                path,
                ref_dataset,
                include_bleurt=not args.no_bleurt,
                include_bertscore=not args.no_bertscore,
            )
            results[name] = metrics
            print(f"    FlagAcc={metrics.get('Error Flags Accuracy',0):.4f}  "
                  f"SentAcc={metrics.get('Error Sentence Detection Accuracy',0):.4f}  "
                  f"ROUGE1={metrics.get('ROUGE1',0):.4f}  "
                  f"BERTScore={metrics.get('BERTSCORE',0):.4f}"
                  + (f"  BLEURT={metrics.get('BLEURT',0):.4f}" if "BLEURT" in metrics else ""))
        except Exception as exc:
            print(f"    ERROR: {exc}")

    # Save raw results JSON
    json_path = run_dir / "ablation_results.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nRaw results saved to: {json_path}")

    # Print & save report
    if results:
        print(f"\n{'='*60}")
        report_path = run_dir / "ablation_report.txt"
        print_and_save_report(results, report_path, args.model, n_samples, args.dataset)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
