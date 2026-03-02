"""
MEDIQA-CORR Modular Pipeline Framework

This module defines the abstract interfaces for building medical error detection
pipelines. Each component is fully modular and can be swapped independently:

    - LLMProvider: wraps any LLM (Gemini, GPT, Claude, etc.)
    - PromptAugmenter: enriches prompts with external knowledge (RxNorm RAG, etc.)
    - Predictor: combines an LLMProvider + optional PromptAugmenter to produce predictions
    - Runner: loads data, runs a Predictor over it, and writes submission files

Submission format (compatible with evaluate.py):
    [text_id] [error_flag] [sentence_id] [corrected_sentence_or_NA]

Example usage:
    from pipeline.base import Runner
    from pipeline.providers import GeminiProvider
    from pipeline.augmenters import RxNormAugmenter

    provider = GeminiProvider(model_name="gemini-2.5-flash")
    augmenter = RxNormAugmenter(extractor_type="pubmedbert")
    runner = Runner(provider=provider, augmenter=augmenter)
    runner.run(output_path="outputs/my_run.txt", split="test")
"""

from __future__ import annotations

import json
import re
import time
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset

logger = logging.getLogger(__name__)

# Suppress noisy HTTP logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


# =============================================================================
# Data Types
# =============================================================================

@dataclass
class MedicalTextItem:
    """A single clinical text item from the MEDEC dataset."""
    text_id: str
    text: str
    sentences: str
    # Ground truth (optional, for evaluation)
    error_flag: Optional[bool] = None
    error_type: Optional[str] = None
    error_sentence_id: Optional[int] = None
    corrected_sentence: Optional[str] = None


@dataclass
class Prediction:
    """A single prediction output — the universal exchange format between components."""
    text_id: str
    error_flag: int           # 0 = no error, 1 = error detected
    sentence_id: int          # -1 if no error, else the error sentence ID
    corrected_sentence: str   # "NA" if no error, else the corrected sentence
    metadata: Dict[str, Any] = field(default_factory=dict)  # extra info (extracted drugs, etc.)


# =============================================================================
# Abstract Base Classes
# =============================================================================

class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Implement this to wrap any LLM (Gemini, GPT, Claude, local models, etc.).
    The provider is responsible for sending a prompt and returning raw text output.
    """

    @abstractmethod
    def generate(self, user_prompt: str, system_prompt: str = "") -> str:
        """
        Generate a response from the LLM.

        Args:
            user_prompt: The user/input prompt to send.
            system_prompt: The system/instruction prompt.

        Returns:
            Raw text output from the LLM.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this provider (used in output file names)."""
        ...


class PromptAugmenter(ABC):
    """
    Abstract base class for prompt augmenters.

    Implement this to enrich prompts with external knowledge before
    sending them to the LLM. Examples: RxNorm drug info, SNOMED CT
    terminology, clinical guidelines, etc.
    """

    @abstractmethod
    def augment(
        self,
        system_prompt: str,
        text: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Augment a system prompt with retrieved knowledge.

        Args:
            system_prompt: The original system prompt.
            text: The medical text to analyze for knowledge retrieval.

        Returns:
            Tuple of:
                - augmented_system_prompt: The enriched system prompt.
                - metadata: Dict with retrieval info (e.g. {"extracted_drugs": [...]}).
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this augmenter (e.g., 'rxnorm-pubmedbert')."""
        ...


# =============================================================================
# Predictor
# =============================================================================

# Default system prompt for the MEDIQA-CORR task
DEFAULT_SYSTEM_PROMPT = """\
The following is a medical narrative about a patient. You are a skilled medical \
doctor reviewing the clinical text. The text is either correct or contains one error.
The text has a sentence per line. Each line starts with the sentence ID, followed \
by a pipe character then the sentence to check. Check every sentence of the text.
If the text is correct return the following output: CORRECT. If the text has a \
medical error, return the sentence id of the sentence containing the error, \
followed by a space, and a corrected version of the sentence."""


def parse_model_output(output_text: str) -> Tuple[int, int, str]:
    """
    Parse raw LLM output into submission fields.

    Returns:
        Tuple of (error_flag, sentence_id, corrected_sentence)
    """
    t = (output_text or "").strip()

    if t.upper().rstrip(".") == "CORRECT":
        return 0, -1, "NA"

    parts = t.split(None, 1)
    if not parts:
        return 0, -1, "NA"

    try:
        sid = int(parts[0])
    except ValueError:
        return 0, -1, "NA"

    corrected = parts[1].strip() if len(parts) > 1 else "NA"
    if not corrected:
        corrected = "NA"

    return 1, sid, corrected


class Predictor:
    """
    Combines an LLMProvider with an optional PromptAugmenter to produce
    predictions on medical text items.

    This is the main "inference engine" — it takes a MedicalTextItem,
    optionally augments the prompt, calls the LLM, and parses the output.
    """

    def __init__(
        self,
        provider: LLMProvider,
        augmenter: Optional[PromptAugmenter] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        self.provider = provider
        self.augmenter = augmenter
        self.system_prompt = system_prompt

    def predict(self, item: MedicalTextItem) -> Prediction:
        """Run prediction on a single item."""
        system_prompt = self.system_prompt
        metadata: Dict[str, Any] = {}

        # Augment prompt if we have an augmenter
        if self.augmenter:
            system_prompt, aug_meta = self.augmenter.augment(system_prompt, item.text)
            metadata.update(aug_meta)

        # Call the LLM
        try:
            raw_output = self.provider.generate(
                user_prompt=item.sentences,
                system_prompt=system_prompt
            )
        except Exception as e:
            logger.error(f"LLM error for {item.text_id}: {e}")
            raw_output = "CORRECT"

        metadata["raw_output"] = raw_output

        # Parse into structured prediction
        flag, sid, corrected = parse_model_output(raw_output)

        return Prediction(
            text_id=item.text_id,
            error_flag=flag,
            sentence_id=sid,
            corrected_sentence=corrected,
            metadata=metadata
        )

    @property
    def label(self) -> str:
        """Descriptive label for this predictor configuration."""
        parts = [self.provider.name]
        if self.augmenter:
            parts.append(self.augmenter.name)
        return "_".join(parts)


# =============================================================================
# Submission Formatter (compatible with evaluate.py)
# =============================================================================

def escape_for_submission(s: str) -> str:
    """Escape a corrected sentence for the submission format."""
    if s == "NA":
        return "NA"
    escaped = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def format_submission_line(pred: Prediction) -> str:
    """
    Format a single prediction as a submission line.

    Format: [text_id] [error_flag] [sentence_id] [corrected_sentence_or_NA]
    Compatible with evaluate.py's parse_run_submission_file().
    """
    if pred.error_flag == 0:
        return f"{pred.text_id} 0 -1 NA"
    else:
        return f"{pred.text_id} 1 {pred.sentence_id} {escape_for_submission(pred.corrected_sentence)}"


# =============================================================================
# Dataset Loader
# =============================================================================

def load_medec_dataset(
    split: str = "test",
    dataset_name: str = "mkieffer/MEDEC-MS",
    indices_path: Optional[str] = None,
    num_samples: Optional[int] = None,
) -> List[MedicalTextItem]:
    """
    Load the MEDEC dataset and return as a list of MedicalTextItem.

    Args:
        split: Dataset split ("test", "train", "validation").
        dataset_name: HuggingFace dataset name.
        indices_path: Optional path to a JSON file with {"indices": [...]} for subsetting.
        num_samples: Optional limit on number of samples.

    Returns:
        List of MedicalTextItem objects.
    """
    logger.info(f"Loading dataset: {dataset_name} split={split}")
    dataset = load_dataset(dataset_name, split=split)

    if indices_path:
        path = Path(indices_path)
        if path.exists():
            indices = json.loads(path.read_text(encoding="utf-8"))["indices"]
            dataset = dataset.select(indices)
            logger.info(f"Selected {len(dataset)} samples from indices file")
        else:
            logger.warning(f"Indices file not found: {path}, using full dataset")

    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
        logger.info(f"Limited to {len(dataset)} samples")

    items = []
    for row in dataset:
        items.append(MedicalTextItem(
            text_id=row["text_id"],
            text=row["text"],
            sentences=row["sentences"],
            error_flag=row.get("error_flag"),
            error_type=row.get("error_type"),
            error_sentence_id=row.get("error_sentence_id"),
            corrected_sentence=row.get("corrected_sentence"),
        ))

    logger.info(f"Loaded {len(items)} items")
    return items


# =============================================================================
# Runner
# =============================================================================

class Runner:
    """
    Runs a Predictor over a dataset and writes results as a submission file.

    The output file is directly compatible with evaluate.py.

    Usage:
        runner = Runner(provider=my_provider, augmenter=my_augmenter)
        runner.run(output_path="outputs/results.txt")
    """

    def __init__(
        self,
        provider: LLMProvider,
        augmenter: Optional[PromptAugmenter] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        rate_limit_delay: float = 1.0,
    ):
        self.predictor = Predictor(
            provider=provider,
            augmenter=augmenter,
            system_prompt=system_prompt,
        )
        self.rate_limit_delay = rate_limit_delay

    def run(
        self,
        output_path: str,
        split: str = "test",
        dataset_name: str = "mkieffer/MEDEC-MS",
        indices_path: Optional[str] = None,
        num_samples: Optional[int] = None,
    ) -> List[Prediction]:
        """
        Run the full pipeline and write a submission file.

        Args:
            output_path: Path to write the submission file.
            split: Dataset split.
            dataset_name: HuggingFace dataset name.
            indices_path: Optional indices file for subsetting.
            num_samples: Optional sample limit.

        Returns:
            List of all Prediction objects.
        """
        # Load data
        items = load_medec_dataset(
            split=split,
            dataset_name=dataset_name,
            indices_path=indices_path,
            num_samples=num_samples,
        )

        # Ensure output directory exists
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"{'='*60}")
        print(f"MEDIQA-CORR Pipeline: {self.predictor.label}")
        print(f"{'='*60}")
        print(f"  Samples: {len(items)}")
        print(f"  Output:  {out_path}")
        print()

        predictions: List[Prediction] = []

        with out_path.open("w", encoding="utf-8") as f:
            for i, item in enumerate(items):
                print(f"[{i+1}/{len(items)}] {item.text_id}...", end=" ")

                pred = self.predictor.predict(item)
                predictions.append(pred)

                # Write submission line
                line = format_submission_line(pred)
                f.write(line + "\n")

                drugs = pred.metadata.get("extracted_drugs", [])
                drugs_str = f", drugs={drugs}" if drugs else ""
                print(f"flag={pred.error_flag}, sid={pred.sentence_id}{drugs_str}")

                if i < len(items) - 1:
                    time.sleep(self.rate_limit_delay)

        print(f"\nResults written to: {out_path}")
        print(f"Use evaluate.py to evaluate:")
        print(f"  python evaluation/evaluate.py --submission {out_path}")

        return predictions
