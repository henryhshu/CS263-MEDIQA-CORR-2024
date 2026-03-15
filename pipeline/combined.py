"""
CombinedPredictor — Integrates RAG + ICL + Multi-Agent for each request.

For every clinical text item, all three components contribute:

  1. RAG (Knowledge Retrieval)
       Extracts drug/medication names from the text, looks them up in RxNorm,
       and injects the drug reference block into the Detector and Critic
       system instructions. This grounds the agents in verified drug facts.

  2. ICL (In-Context Learning)
       Retrieves the k most similar training examples via embedding similarity,
       and prepends them as labeled demonstrations to the Detector's user input.
       This shows the model what a good correction looks like for similar cases.

  3. Multi-Agent (Detector → Critic fast-path → Editor → Critic)
       The core reasoning pipeline from multi-agent/multi-agent-detect-critic-edit.py.
       All original agent logic is preserved — only the instructions and inputs
       are enriched by the RAG and ICL steps above.

Nothing in the original modules is modified. This class imports and calls
their public functions directly.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ── Module loaders (same helpers as run_integrated.py) ───────────────────────

def _load_module_from_file(name: str, filepath: Path):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_icl():
    return _load_module_from_file(
        "in_context_learning",
        REPO_ROOT / "in-context-learning" / "in-context-learning.py",
    )


def _load_ma():
    return _load_module_from_file(
        "multi_agent_dce",
        REPO_ROOT / "multi-agent" / "multi-agent-detect-critic-edit.py",
    )


# ── ICL example formatter ────────────────────────────────────────────────────

def _build_icl_prefix(examples: List[Dict[str, Any]]) -> str:
    """
    Format retrieved training examples as labeled demonstrations for the Detector.

    Each example shows the sentence block and the expected answer, so the model
    can learn the correction pattern from similar cases before seeing the query.
    """
    if not examples:
        return ""
    lines = ["SIMILAR CASES FOR REFERENCE (use these to calibrate your answer):"]
    for i, ex in enumerate(examples, start=1):
        lines.append(f"\n--- Example {i} ---")
        lines.append(ex["sentences"])
        if ex["error_flag"] == 0:
            lines.append("CORRECT ANSWER: CORRECT.")
        else:
            lines.append(
                f"CORRECT ANSWER: {ex['error_sentence_id']} {ex['corrected_sentence']}"
            )
    lines.append("\n--- Now evaluate the following case ---")
    return "\n".join(lines) + "\n\n"


# ── CombinedPredictor ─────────────────────────────────────────────────────────

class CombinedPredictor:
    """
    Configurable predictor that can combine any subset of RAG, ICL, and Multi-Agent.

    All three enabled → full integration (RAG + ICL + Multi-Agent per item).
    Disable any component to produce ablation variants:

      use_rag=False, use_icl=True,  use_multiagent=True  → ICL + MA  (no RAG)
      use_rag=True,  use_icl=False, use_multiagent=True  → RAG + MA  (no ICL)
      use_rag=True,  use_icl=True,  use_multiagent=False → RAG + ICL (single-pass)
      use_rag=False, use_icl=False, use_multiagent=True  → MA only
      ... etc.

    Usage:
        predictor = CombinedPredictor(model="gpt-4.1", k_shot=5)
        predictor.load_train_data(train_items)   # embed once, reuse across predictions
        line = predictor.predict(row)            # submission-format string
    """

    def __init__(
        self,
        model: str = "gpt-4.1",
        k_shot: int = 5,
        rag_extractor_type: str = "pubmedbert",
        embedding_model: str = "text-embedding-3-small",
        n_best: int = 3,
        use_rag: bool = True,
        use_icl: bool = True,
        use_multiagent: bool = True,
    ):
        self.model = model
        self.k_shot = k_shot
        self.embedding_model = embedding_model
        self.n_best = n_best
        self.use_rag = use_rag
        self.use_icl = use_icl
        self.use_multiagent = use_multiagent

        # Load original modules (unmodified)
        self._icl = _load_icl()
        self._ma = _load_ma()

        # RAG: RxNorm drug context (from knowledge_retrieval/)
        if use_rag:
            from knowledge_retrieval.rxnorm_rag import RxNormRAGContext
            self._rag = RxNormRAGContext(extractor_type=rag_extractor_type)
        else:
            self._rag = None

        # OpenAI client (the multi-agent module has already set OPENAI_API_KEY)
        from openai import OpenAI
        self._client = OpenAI()

        # Multi-agent Config (used when use_multiagent=True)
        self._cfg = self._ma.Config(
            detector_model=model,
            critic_model=model,
            editor_model=model,
            n_best=n_best,
        )

        self._train_items: Optional[List[Dict[str, Any]]] = None
        self._train_embeddings: Optional[Dict[str, List[float]]] = None

    @property
    def label(self) -> str:
        """Short label for this ablation configuration."""
        parts = []
        if self.use_rag:
            parts.append("rag")
        if self.use_icl:
            parts.append(f"icl_k{self.k_shot}")
        if self.use_multiagent:
            parts.append("ma")
        return "+".join(parts) if parts else "none"

    def load_train_data(self, train_items: List[Dict[str, Any]]) -> None:
        """
        Embed training items for dynamic ICL retrieval.
        Called once before running predictions — embeddings are cached to disk.
        """
        self._train_items = train_items
        cache_dir = REPO_ROOT / "in-context-learning" / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Embedding {len(train_items)} training items for ICL retrieval...")
        self._train_embeddings = self._icl.embed_items(
            train_items,
            self.embedding_model,
            cache_dir,
            "train",
        )
        print("  Done.")

    def predict(self, row: Dict[str, Any]) -> str:
        """
        Run RAG + ICL + Multi-Agent for a single dataset row.

        Args:
            row: Dict with at least 'text_id', 'sentences', and 'text' fields
                 (standard MEDEC/MEDEC-MS row format).

        Returns:
            A submission-format string:  "<text_id> <flag> <sid> <correction>"
        """
        ma = self._ma
        icl = self._icl

        text_id = row["text_id"]
        sentences_text = row["sentences"]
        full_text = row.get("text", sentences_text)

        sents = ma.parse_sentences_field(sentences_text)
        sentences_block = ma.to_sentences_block(sents)
        valid_ids = {x["id"] for x in sents}
        id2sent = {x["id"]: x["sentence"] for x in sents}

        # ── Step 1: RAG — build drug-enriched instructions ───────────────────
        # For multi-agent mode, RAG augments the structured DETECTOR/CRITIC prompts.
        # For single-pass mode, RAG augments DEFAULT_SYSTEM_PROMPT (plain-text output)
        # so the response can be parsed by the standard plain-text parser.
        from pipeline.base import DEFAULT_SYSTEM_PROMPT
        if self.use_rag and self._rag is not None:
            detector_instructions, _, _ = self._rag.build_augmented_prompt(
                original_text=full_text,
                system_prompt=ma.DETECTOR_INSTRUCTIONS,
                include_context=True,
                max_drugs=10,
            )
            critic_instructions, _, _ = self._rag.build_augmented_prompt(
                original_text=full_text,
                system_prompt=ma.CRITIC_INSTRUCTIONS,
                include_context=True,
                max_drugs=10,
            )
            single_pass_instructions, _, _ = self._rag.build_augmented_prompt(
                original_text=full_text,
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                include_context=True,
                max_drugs=10,
            )
        else:
            detector_instructions = ma.DETECTOR_INSTRUCTIONS
            critic_instructions = ma.CRITIC_INSTRUCTIONS
            single_pass_instructions = DEFAULT_SYSTEM_PROMPT
        editor_instructions = ma.EDITOR_INSTRUCTIONS  # editor rewrites, not diagnoses

        # ── Step 2: ICL — retrieve similar training examples ──────────────────
        examples: List[Dict[str, Any]] = []
        if self.use_icl and self._train_items and self.k_shot > 0:
            cache_dir = REPO_ROOT / "in-context-learning" / "cache"
            query_text = icl.retrieval_text(icl.canonicalize_item(dict(row)))
            query_embedding = icl.cached_embedding(
                cache_dir=cache_dir,
                namespace="query",
                model=self.embedding_model,
                cache_id=text_id,
                text=query_text,
            )
            examples = icl.nearest_examples(
                self._train_items,
                self._train_embeddings,
                query_embedding,
                self.k_shot,
            )

        # ── Step 3a: Single-pass mode (use_multiagent=False) ──────────────────
        # RAG-enriched system prompt + ICL examples as conversation history,
        # then one LLM call (same format as standard ICL, but with drug context).
        if not self.use_multiagent:
            messages = [{"role": "system", "content": single_pass_instructions}]
            for ex in examples:
                messages.append({"role": "user", "content": ex["sentences"]})
                messages.append({"role": "assistant", "content": icl.gold_response(ex)})
            messages.append({"role": "user", "content": sentences_block})
            output = icl.call_openai_response(self.model, messages)
            flag, sid, corrected = icl.parse_model_output(output)
            return ma.to_submission_line(
                text_id, flag, sid, corrected if corrected else "NA"
            )

        # ── Step 3b: Multi-Agent mode ─────────────────────────────────────────
        icl_prefix = _build_icl_prefix(examples)
        detector_input = icl_prefix + ma.detector_prompt(sentences_block)

        det: "ma.DetectorOut" = ma.call_parse(
            self._client,
            self.model,
            detector_instructions,
            detector_input,
            ma.DetectorOut,
        )

        if det.verdict == "CORRECT" or det.error_sentence_id == -1:
            return ma.to_submission_line(text_id, 0, -1, "NA")

        sid = det.error_sentence_id
        if sid not in valid_ids:
            return ma.to_submission_line(text_id, 0, -1, "NA")

        corrected = (det.corrected_sentence or "").strip()
        if not corrected or corrected == "NA":
            corrected = ""

        # Fast path: Critic checks Detector's correction first
        if corrected:
            single = ma.make_single_proposal(sid, corrected)
            critic1: "ma.CriticOut" = ma.call_parse(
                self._client,
                self.model,
                critic_instructions,
                ma.critic_prompt(sentences_block, single),
                ma.CriticOut,
            )
            if critic1.overall_recommendation == "choose_rank_1":
                return ma.to_submission_line(text_id, 1, sid, corrected)

        # Full path: Editor generates n-best, Critic selects
        editor_out: "ma.EditorOut" = ma.call_parse(
            self._client,
            self.model,
            editor_instructions,
            ma.editor_prompt(sentences_block, sid, id2sent[sid], n=self.n_best),
            ma.EditorOut,
        )
        critic2: "ma.CriticOut" = ma.call_parse(
            self._client,
            self.model,
            critic_instructions,
            ma.critic_prompt(sentences_block, editor_out),
            ma.CriticOut,
        )
        final = ma.pick_recommended(editor_out, critic2)
        if not final:
            final = "NA"

        return ma.to_submission_line(text_id, 1, sid, final)
