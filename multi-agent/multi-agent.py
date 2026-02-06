"""
LLM Multi-Agent Orchestration Skeleton (Python, OpenAI Responses API)

Pipeline:
  Detector -> (if error or uncertain) Localizer -> Corrector -> Reviewer/Judge -> Final

Key features:
  - Role-based multi-call orchestration (a single LLM is enough)
  - Structured Outputs with JSON Schema (strict) enforced by the API
  - Pydantic validation (defense-in-depth)
  - Span validation against the original text
  - JSONL logging for every step

Install:
  pip install openai pydantic

Env:
  export OPENAI_API_KEY="..."
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from pathlib import Path

def load_api_key_from_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"API key file not found: {path}")
    key = path.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError(f"API key file is empty: {path}")
    return key

key_path = Path.home() / "env" / "openai_secret_key.txt"

OPENAI_API_KEY = load_api_key_from_file(key_path)

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# =========================
# 0) OpenAI client
# =========================
client = OpenAI()


# =========================
# 1) Types and Schemas (Pydantic)
# =========================

ErrorType = Literal[
    "MedicationName",
    "Dose",
    "Route",
    "Frequency",
    "AllergyConflict",
    "LabValue",
    "DiagnosisMismatch",
    "NegationError",
    "TemporalInconsistency",
    "Other",
]
Decision = Literal["ACCEPT", "REVISE", "HUMAN_REVIEW"]


class DetectorOut(BaseModel):
    has_error: bool
    certainty: float = Field(ge=0.0, le=1.0)
    suspected_types: List[ErrorType] = Field(default_factory=list)


class Span(BaseModel):
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    text: str
    type: ErrorType = "Other"


class LocalizerOut(BaseModel):
    spans: List[Span] = Field(default_factory=list)


class CorrectorOut(BaseModel):
    proposed_correction: str
    edit_scope: Literal["span", "sentence", "snippet"] = "sentence"
    notes: Optional[str] = None


class FinalOut(BaseModel):
    has_error: bool
    error_spans: List[Span] = Field(default_factory=list)
    correction: Optional[str] = None
    rationale_short: Optional[str] = None
    certainty: float = Field(ge=0.0, le=1.0)
    needs_human_review: bool = False


class ReviewerOut(BaseModel):
    decision: Decision
    final: FinalOut
    issues: List[str] = Field(default_factory=list)


# =========================
# 2) Agent instructions (developer messages)
# =========================

DETECTOR_SYSTEM = """You are Agent A (Detector).
Task: Decide if the clinical text contains a medical error.
Return ONLY JSON that matches the enforced schema. Do not add keys.
Rules:
- If unclear or insufficient context, prefer has_error=false or lower certainty.
"""

LOCALIZER_SYSTEM = """You are Agent B (Localizer).
Task: Identify the error text span(s) in the given clinical text IF an error exists.
Return ONLY JSON that matches the enforced schema. Do not add keys.
Rules:
- "text" MUST be an exact substring from the original input.
- start/end are character offsets in the original text (Python slicing: text[start:end]).
- Provide 1 to 3 spans max.
"""

CORRECTOR_SYSTEM = """You are Agent C (Corrector).
Task: Propose a minimal correction for the given error span within context.
Return ONLY JSON that matches the enforced schema. Do not add keys.
Rules:
- Minimal edit: change as little as necessary.
- Do NOT add new medical facts not supported by the text.
- If correction cannot be made safely, set proposed_correction="" and explain briefly in notes.
"""

REVIEWER_SYSTEM = """You are Agent D (Reviewer/Judge).
Task: Verify consistency, avoid over-correction/hallucination, and produce the final decision.
Return ONLY JSON that matches the enforced schema. Do not add keys.
Rules:
- If evidence is insufficient or risk is high, choose HUMAN_REVIEW and needs_human_review=true.
- Ensure span text matches the original exactly.
- Keep rationale_short concise.
"""


# =========================
# 3) JSON Schemas for Structured Outputs (strict)
# =========================
# Note: Keep schemas within supported JSON Schema features. Prefer:
# - type/object/array/string/number/boolean
# - enum
# - required
# - additionalProperties: false
# - minItems/maxItems, minimum/maximum
# See OpenAI Structured Outputs docs for supported subsets.

ERROR_TYPE_ENUM = [
    "MedicationName",
    "Dose",
    "Route",
    "Frequency",
    "AllergyConflict",
    "LabValue",
    "DiagnosisMismatch",
    "NegationError",
    "TemporalInconsistency",
    "Other",
]

SPAN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "start": {"type": "integer", "minimum": 0},
        "end": {"type": "integer", "minimum": 0},
        "text": {"type": "string"},
        "type": {"type": "string", "enum": ERROR_TYPE_ENUM},
    },
    "required": ["start", "end", "text", "type"],
}

DETECTOR_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "has_error": {"type": "boolean"},
        "certainty": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "suspected_types": {
            "type": "array",
            "items": {"type": "string", "enum": ERROR_TYPE_ENUM},
        },
    },
    "required": ["has_error", "certainty", "suspected_types"],
}

LOCALIZER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "spans": {
            "type": "array",
            "minItems": 0,
            "maxItems": 3,
            "items": SPAN_SCHEMA,
        }
    },
    "required": ["spans"],
}

CORRECTOR_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "proposed_correction": {"type": "string"},
        "edit_scope": {"type": "string", "enum": ["span", "sentence", "snippet"]},
        "notes": {"type": ["string", "null"]},
    },
    "required": ["proposed_correction", "edit_scope", "notes"],
}

FINAL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "has_error": {"type": "boolean"},
        "error_spans": {"type": "array", "items": SPAN_SCHEMA},
        "correction": {"type": ["string", "null"]},
        "rationale_short": {"type": ["string", "null"]},
        "certainty": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "needs_human_review": {"type": "boolean"},
    },
    "required": [
        "has_error",
        "error_spans",
        "correction",
        "rationale_short",
        "certainty",
        "needs_human_review",
    ],
}

REVIEWER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decision": {"type": "string", "enum": ["ACCEPT", "REVISE", "HUMAN_REVIEW"]},
        "final": FINAL_SCHEMA,
        "issues": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["decision", "final", "issues"],
}


def json_schema_format(name: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """Build the `text.format` payload for Responses API Structured Outputs."""
    return {
        "type": "json_schema",
        "name": name,
        "strict": True,
        "schema": schema,
    }


# =========================
# 4) Provider wrapper (OpenAI Responses API with strict JSON schema)
# =========================

@dataclass
class LLMConfig:
    # IMPORTANT: Structured Outputs via json_schema requires compatible model snapshots.
    # For example: "gpt-4o-2024-08-06" or later.
    model: str = "gpt-4o-2024-08-06"
    temperature: float = 0.0
    max_output_tokens: int = 800


def call_llm(
    system_prompt: str,
    user_prompt: str,
    cfg: LLMConfig,
    schema_name: str,
    schema: Dict[str, Any],
) -> str:
    """Call the model and force outputs to match the given JSON schema (strict)."""
    resp = client.responses.create(
        model=cfg.model,
        input=[
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=cfg.temperature,
        max_output_tokens=cfg.max_output_tokens,
        text={"format": json_schema_format(schema_name, schema)},
    )
    # With Structured Outputs, this should be a single JSON object string.
    return resp.output_text.strip()


def parse_strict(
    schema_model: type[BaseModel],
    system_prompt: str,
    user_prompt: str,
    cfg: LLMConfig,
    schema_name: str,
    schema_dict: Dict[str, Any],
) -> BaseModel:
    """
    Parse the model output into a Pydantic model.
    In strict json_schema mode, retries are usually unnecessary, but we keep
    Pydantic validation as defense-in-depth.
    """
    raw = call_llm(system_prompt, user_prompt, cfg, schema_name, schema_dict)
    try:
        return schema_model.model_validate_json(raw)
    except (ValidationError, json.JSONDecodeError) as e:
        # If this happens, log/inspect raw. You can optionally retry here.
        raise RuntimeError(f"Schema parse failed for {schema_name}: {e}\nRaw:\n{raw}") from e


# =========================
# 5) Utilities: span validation and logging
# =========================

def validate_span(text: str, span: Span) -> Tuple[bool, str]:
    if span.start < 0 or span.end < 0 or span.start > span.end or span.end > len(text):
        return False, "span_out_of_bounds"
    if text[span.start:span.end] != span.text:
        return False, "span_text_mismatch"
    return True, ""


class JSONLLogger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def log(self, record: Dict[str, Any]) -> None:
        record = dict(record)
        record["ts"] = time.time()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# =========================
# 6) Orchestrator
# =========================

class Orchestrator:
    def __init__(self, cfg: LLMConfig, log_path: str = "logs/med_error_runs.jsonl"):
        self.cfg = cfg
        self.logger = JSONLLogger(log_path)

    def run(self, clinical_text: str) -> FinalOut:
        run_id = str(uuid.uuid4())
        self.logger.log({"run_id": run_id, "step": "input", "clinical_text": clinical_text})

        # A) Detector
        det: DetectorOut = parse_strict(
            DetectorOut,
            DETECTOR_SYSTEM,
            f"Clinical text:\n{clinical_text}",
            self.cfg,
            schema_name="detector_out",
            schema_dict=DETECTOR_SCHEMA,
        )  # type: ignore
        self.logger.log({"run_id": run_id, "step": "detector", "out": det.model_dump()})

        # Early exit if confidently NO_ERROR
        if not det.has_error and det.certainty >= 0.85:
            final = FinalOut(
                has_error=False,
                error_spans=[],
                correction=None,
                rationale_short=None,
                certainty=det.certainty,
                needs_human_review=False,
            )
            self.logger.log({"run_id": run_id, "step": "final", "out": final.model_dump()})
            return final

        # B) Localizer
        loc: LocalizerOut = parse_strict(
            LocalizerOut,
            LOCALIZER_SYSTEM,
            "Clinical text:\n"
            + clinical_text
            + "\n\nDetector output:\n"
            + det.model_dump_json(),
            self.cfg,
            schema_name="localizer_out",
            schema_dict=LOCALIZER_SCHEMA,
        )  # type: ignore
        self.logger.log({"run_id": run_id, "step": "localizer_raw", "out": loc.model_dump()})

        valid_spans: List[Span] = []
        span_issues: List[str] = []
        for sp in loc.spans[:3]:
            ok, reason = validate_span(clinical_text, sp)
            if ok:
                valid_spans.append(sp)
            else:
                span_issues.append(reason)

        # If localization fails, route to human review
        if not valid_spans:
            final = FinalOut(
                has_error=det.has_error,
                error_spans=[],
                correction=None,
                rationale_short="Span localization failed; route to human review.",
                certainty=min(det.certainty, 0.5),
                needs_human_review=True,
            )
            self.logger.log({"run_id": run_id, "step": "final", "out": final.model_dump(), "issues": span_issues})
            return final

        # C) Corrector (for each candidate span)
        corrections: List[Dict[str, Any]] = []
        for i, sp in enumerate(valid_spans, start=1):
            cor: CorrectorOut = parse_strict(
                CorrectorOut,
                CORRECTOR_SYSTEM,
                f"Clinical text:\n{clinical_text}\n\nSpan candidate:\n{sp.model_dump_json()}",
                self.cfg,
                schema_name=f"corrector_out_{i}",
                schema_dict=CORRECTOR_SCHEMA,
            )  # type: ignore
            corrections.append({"span": sp.model_dump(), "correction": cor.model_dump()})
            self.logger.log(
                {"run_id": run_id, "step": f"corrector_{i}", "span": sp.model_dump(), "out": cor.model_dump()}
            )

        # D) Reviewer/Judge
        rev: ReviewerOut = parse_strict(
            ReviewerOut,
            REVIEWER_SYSTEM,
            "Clinical text:\n"
            + clinical_text
            + "\n\nDetector:\n"
            + det.model_dump_json()
            + "\n\nSpan candidates:\n"
            + json.dumps([s.model_dump() for s in valid_spans], ensure_ascii=False)
            + "\n\nCorrections:\n"
            + json.dumps(corrections, ensure_ascii=False),
            self.cfg,
            schema_name="reviewer_out",
            schema_dict=REVIEWER_SCHEMA,
        )  # type: ignore
        self.logger.log({"run_id": run_id, "step": "reviewer", "out": rev.model_dump()})

        # Final sanity check: validate reviewer spans again
        final_issues: List[str] = list(rev.issues)
        for sp in rev.final.error_spans:
            ok, reason = validate_span(clinical_text, sp)
            if not ok:
                final_issues.append(reason)
                rev.final.needs_human_review = True
                rev.final.rationale_short = ((rev.final.rationale_short or "").strip() + f" (Span check: {reason})").strip()
                rev.decision = "HUMAN_REVIEW"

        self.logger.log(
            {
                "run_id": run_id,
                "step": "final",
                "decision": rev.decision,
                "issues": final_issues,
                "out": rev.final.model_dump(),
            }
        )
        return rev.final


# =========================
# 7) Example usage
# =========================

if __name__ == "__main__":
    cfg = LLMConfig(
        model="gpt-4o-2024-08-06",  # Use a compatible snapshot for json_schema structured outputs.
        temperature=0.0,
        max_output_tokens=800,
    )
    orch = Orchestrator(cfg)

    sample_text = (
        "Patient is a 65-year-old male with type 2 diabetes. "
        "Continue metformin 5000 mg daily. Denies allergies."
    )

    result = orch.run(sample_text)
    print(result.model_dump_json(indent=2, ensure_ascii=False))
