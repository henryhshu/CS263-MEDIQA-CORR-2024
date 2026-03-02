#!/usr/bin/env python
"""
MEDIQA-CORR Pipeline Runner

Unified CLI for running medical error detection experiments with any
combination of LLM provider and prompt augmenter.

Usage:
    # Baseline (no augmentation)
    python -m pipeline.run --provider gemini --output outputs/baseline.txt

    # With RxNorm RAG (PubMedBERT NER)
    python -m pipeline.run --provider gemini --augmenter rxnorm-pubmedbert --output outputs/rag.txt

    # With RxNorm RAG (regex, legacy)
    python -m pipeline.run --provider gemini --augmenter rxnorm-regex --output outputs/rag_regex.txt

    # OpenAI GPT (for teammate compatibility)
    python -m pipeline.run --provider openai --model gpt-4o --augmenter rxnorm-pubmedbert --output outputs/gpt_rag.txt

    # Full dataset
    python -m pipeline.run --provider gemini --augmenter rxnorm-pubmedbert --full --output outputs/full_rag.txt

    # Subsample
    python -m pipeline.run --provider gemini -n 20 --output outputs/test.txt

After running, evaluate with:
    python evaluation/evaluate.py --submission outputs/baseline.txt
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from pipeline.base import Runner
from pipeline.providers import GeminiProvider, OpenAIProvider
from pipeline.augmenters import NullAugmenter, RxNormAugmenter


INDICES_PATH = str(Path(__file__).parent.parent / "baseline-experiment" / "sampled_test_indices.json")


def build_provider(args):
    """Construct the LLM provider from CLI args."""
    if args.provider == "gemini":
        return GeminiProvider(model_name=args.model)
    elif args.provider == "openai":
        return OpenAIProvider(model_name=args.model)
    else:
        raise ValueError(f"Unknown provider: {args.provider}")


def build_augmenter(args):
    """Construct the prompt augmenter from CLI args."""
    aug = args.augmenter
    if aug is None or aug == "none":
        return None
    elif aug == "rxnorm-pubmedbert":
        return RxNormAugmenter(extractor_type="pubmedbert")
    elif aug == "rxnorm-regex":
        return RxNormAugmenter(extractor_type="regex")
    else:
        raise ValueError(f"Unknown augmenter: {aug}")


def main():
    parser = argparse.ArgumentParser(
        description="MEDIQA-CORR Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Provider
    parser.add_argument(
        "--provider", "-p",
        type=str, default="gemini",
        choices=["gemini", "openai"],
        help="LLM provider (default: gemini)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str, default=None,
        help="Model name. Default: gemini-2.5-flash for gemini, gpt-4o for openai",
    )

    # Augmenter
    parser.add_argument(
        "--augmenter", "-a",
        type=str, default=None,
        choices=["none", "rxnorm-pubmedbert", "rxnorm-regex"],
        help="Prompt augmenter. Default: none (baseline)",
    )

    # Dataset
    parser.add_argument(
        "--full", action="store_true",
        help="Run on the full test set instead of the sampled subset",
    )
    parser.add_argument(
        "--num_samples", "-n",
        type=int, default=None,
        help="Limit number of samples (default: all in selected set)",
    )
    parser.add_argument(
        "--indices",
        type=str, default=INDICES_PATH,
        help="Path to sampled indices JSON file",
    )
    parser.add_argument(
        "--split",
        type=str, default="test",
        help="Dataset split (default: test)",
    )

    # Output
    parser.add_argument(
        "--output", "-o",
        type=str, default=None,
        help="Output submission file path. Auto-generated if not specified.",
    )

    # Run config
    parser.add_argument(
        "--rate_limit", "-r",
        type=float, default=1.0,
        help="Delay between API calls in seconds (default: 1.0)",
    )

    args = parser.parse_args()

    # Default model names
    if args.model is None:
        defaults = {"gemini": "gemini-2.5-flash", "openai": "gpt-4o"}
        args.model = defaults[args.provider]

    # Build components
    provider = build_provider(args)
    augmenter = build_augmenter(args)

    # Auto-generate output path
    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        aug_label = augmenter.name if augmenter else "baseline"
        dataset_label = "full" if args.full else "sampled"
        filename = f"medec-ms_{args.model}_{aug_label}_{dataset_label}_{ts}.txt"
        args.output = str(Path("pipeline/outputs") / filename)

    # Determine indices
    indices_path = None if args.full else args.indices

    # Create and run
    runner = Runner(
        provider=provider,
        augmenter=augmenter,
        rate_limit_delay=args.rate_limit,
    )

    predictions = runner.run(
        output_path=args.output,
        split=args.split,
        indices_path=indices_path,
        num_samples=args.num_samples,
    )

    print(f"\nDone! {len(predictions)} predictions written to {args.output}")


if __name__ == "__main__":
    main()
