"""
Concrete PromptAugmenter Implementations.

Currently includes:
    - RxNormAugmenter: Drug-focused RAG using RxNorm API + NER extraction.
    - NullAugmenter: Pass-through (no augmentation) for baseline comparisons.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

from pipeline.base import PromptAugmenter

logger = logging.getLogger(__name__)


class NullAugmenter(PromptAugmenter):
    """
    No-op augmenter — returns the system prompt unchanged.

    Useful for running baseline comparisons against augmented pipelines.

    Usage:
        augmenter = NullAugmenter()
        prompt, meta = augmenter.augment(system_prompt, text)
        # prompt == system_prompt, meta == {}
    """

    def augment(
        self,
        system_prompt: str,
        text: str,
    ) -> Tuple[str, Dict[str, Any]]:
        return system_prompt, {}

    @property
    def name(self) -> str:
        return "none"


class RxNormAugmenter(PromptAugmenter):
    """
    Augments prompts with drug information from the RxNorm API.

    Extracts drug names from the clinical text (using PubMedBERT NER or regex),
    validates them against RxNorm, and appends relevant drug reference info
    to the system prompt.

    This component is model-agnostic — it produces the augmented prompt, which
    can then be fed to any LLM (Gemini, GPT, Claude, etc.).

    Usage:
        augmenter = RxNormAugmenter(extractor_type="pubmedbert")
        prompt, meta = augmenter.augment(system_prompt, clinical_text)
        # prompt now contains drug reference info
        # meta = {"extracted_drugs": ["metformin", "lisinopril"]}
    """

    def __init__(
        self,
        extractor_type: str = "pubmedbert",
        max_drugs: int = 10,
        **extractor_kwargs,
    ):
        """
        Args:
            extractor_type: "pubmedbert" (NER model) or "regex" (legacy).
            max_drugs: Maximum number of drugs to look up.
            **extractor_kwargs: Extra kwargs passed to the extractor (e.g. confidence_threshold).
        """
        from knowledge_retrieval.rxnorm_rag import RxNormRAGContext

        self._rag = RxNormRAGContext(
            extractor_type=extractor_type,
            **extractor_kwargs,
        )
        self._extractor_type = extractor_type
        self._max_drugs = max_drugs
        logger.info(f"RxNormAugmenter initialized: extractor={extractor_type}")

    def augment(
        self,
        system_prompt: str,
        text: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Extract drugs from the text, look them up in RxNorm, and append
        reference information to the system prompt.
        """
        augmented_system, _, extracted_drugs = self._rag.build_augmented_prompt(
            original_text=text,
            system_prompt=system_prompt,
            include_context=True,
            max_drugs=self._max_drugs,
        )

        metadata = {
            "extracted_drugs": extracted_drugs,
            "extractor_type": self._extractor_type,
        }

        return augmented_system, metadata

    @property
    def name(self) -> str:
        return f"rxnorm-{self._extractor_type}"
