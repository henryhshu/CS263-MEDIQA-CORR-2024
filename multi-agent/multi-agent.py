"""
LLM Multi-Agent Orchestration Skeleton (Python, OpenAI Responses API)
UPDATED for the given task format:

Task:
  (a) Predict Error Flag (1/0)
  (b) If Error Flag = 1: Extract Error Sentence ID (integer index)
  (c) If Error Flag = 1: Generate Corrected Sentence (free-text)

Dataset format (important):
  - Each text has a "Sentences" field where each sentence is prefixed by its sentence ID:
      "0 ...\n1 ...\n2 ..."

Submission format (per line):
  [Text ID] [Error Flag] [Error sentence ID or -1] [Corrected sentence or NA]

Example:
  text-id-1 0 -1 NA
  text-id-2 1 8 "correction of sentence 8..."

Key features:
  - Role-based orchestration (single model, multiple calls)
  - Structured Outputs: strict JSON Schema enforced by the API
  - Pydantic validation (defense in depth)
  - Sentence ID validation against provided sentence IDs
  - JSONL logging

Install:
  pip install openai pydantic

Env:
  export OPENAI_API_KEY="..."
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple
from datetime import datetime

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
from datasets import load_dataset

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

client = OpenAI()

# =========================
# 1) Schemas (Pydantic)
# =========================

Decision = Literal["ACCEPT", "REVISE", "HUMAN_REVIEW"]


class DetectorOut(BaseModel):
    error_flag: int = Field(ge=0, le=1)
    certainty: float = Field(ge=0.0, le=1.0)


class LocalizerOut(BaseModel):
    # For no-error texts, must be -1
    error_sentence_id: int
    certainty: float = Field(ge=0.0, le=1.0)


class CorrectorOut(BaseModel):
    # For error_flag=1: corrected sentence text only (not full note)
    corrected_sentence: str
    notes: Optional[str] = None


class FinalOut(BaseModel):
    text_id: str
    error_flag: int = Field(ge=0, le=1)
    error_sentence_id: int
    corrected_sentence: str  # "NA" if error_flag=0
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
Task: Predict whether the clinical text contains a medical error.
Return ONLY JSON matching the enforced schema.
Rules:
- Output error_flag: 1 if there is an error, else 0.
- If uncertain, use lower certainty.
"""

# LOCALIZER_SYSTEM = """You are Agent B (Localizer).
# Task: If error_flag=1, identify the sentence ID that contains the error using the provided sentence list.
# Return ONLY JSON matching the enforced schema.
# Rules:
# - error_sentence_id must be one of the provided IDs when error_flag=1.
# - If error_flag=0, return error_sentence_id=-1.
# - Do NOT return character spans. Sentence ID only.
# """

# LOCALIZER_SYSTEM = """You are Agent B (Localizer).
# Task: If error_flag=1, identify the sentence ID that contains the error using the provided sentence list.
# Return ONLY JSON matching the enforced schema.
# Rules:
# - The text has ONE sentence per line. Each line starts with the sentence ID, followed by a pipe character ('|'), then the sentence to check.
# - Check EVERY sentence line in the text before deciding.
# - When error_flag=1, error_sentence_id MUST be exactly one of the provided sentence IDs.
# - When error_flag=0, return error_sentence_id = -1.
# - Do NOT return character spans. Sentence ID only.
# """

LOCALIZER_SYSTEM = """You are Agent B (Localizer).
Task: If error_flag=1, identify the sentence ID that contains the error using the provided sentence list.
Return ONLY JSON matching the enforced schema.
Rules:
- The text has ONE sentence per line. Each line starts with the sentence ID, followed by a pipe character ('|'), then the sentence to check.
- Sentence IDs start from 0 (zero-based indexing).
- You MUST return the EXACT sentence ID number as written in the input (do NOT renumber, do NOT count lines).
- Check EVERY sentence line in the text before deciding.
- When error_flag=1, error_sentence_id MUST be exactly one of the provided sentence IDs.
- When error_flag=0, return error_sentence_id = -1.
- Do NOT return character spans. Sentence ID only.
"""

CORRECTOR_SYSTEM = """You are Agent C (Corrector).
Task: If error_flag=1, generate the corrected version of ONLY the erroneous sentence (not the whole text).
Return ONLY JSON matching the enforced schema.
Rules:
- Keep edits minimal.
- Do NOT add new facts beyond what is required to correct the error.
- corrected_sentence should be a single sentence (no leading sentence ID).
"""

REVIEWER_SYSTEM = """You are Agent D (Reviewer/Judge).
Task: Ensure the output matches the task rules and produce the final submission fields.
Return ONLY JSON matching the enforced schema.
Rules:
- If error_flag=0: error_sentence_id must be -1 and corrected_sentence must be "NA".
- If error_flag=1: error_sentence_id must be a valid provided ID and corrected_sentence must NOT be "NA".
- If validation is unclear, set needs_human_review=true and choose HUMAN_REVIEW.
"""


# =========================
# 3) JSON Schemas for Structured Outputs (strict)
# =========================

DETECTOR_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "error_flag": {"type": "integer", "enum": [0, 1]},
        "certainty": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["error_flag", "certainty"],
}

LOCALIZER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "error_sentence_id": {"type": "integer"},
        "certainty": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["error_sentence_id", "certainty"],
}

CORRECTOR_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "corrected_sentence": {"type": "string"},
        "notes": {"type": ["string", "null"]},
    },
    "required": ["corrected_sentence", "notes"],
}

FINAL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "text_id": {"type": "string"},
        "error_flag": {"type": "integer", "enum": [0, 1]},
        "error_sentence_id": {"type": "integer"},
        "corrected_sentence": {"type": "string"},
        "certainty": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "needs_human_review": {"type": "boolean"},
    },
    "required": ["text_id", "error_flag", "error_sentence_id", "corrected_sentence", "certainty", "needs_human_review"],
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
    return {"type": "json_schema", "name": name, "strict": True, "schema": schema}


# =========================
# 4) Provider wrapper (OpenAI Responses API, strict JSON Schema)
# =========================

@dataclass
class LLMConfig:
    # Use a model snapshot that supports Structured Outputs (json_schema strict).
    model: str = "gpt-4o-2024-08-06"
    temperature: float = 0.0
    max_output_tokens: int = 600


def call_llm(system_prompt: str, user_prompt: str, cfg: LLMConfig, schema_name: str, schema: Dict[str, Any]) -> str:
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
    return resp.output_text.strip()


def parse_strict(
    schema_model: type[BaseModel],
    system_prompt: str,
    user_prompt: str,
    cfg: LLMConfig,
    schema_name: str,
    schema_dict: Dict[str, Any],
) -> BaseModel:
    raw = call_llm(system_prompt, user_prompt, cfg, schema_name, schema_dict)
    try:
        return schema_model.model_validate_json(raw)
    except (ValidationError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Schema parse failed for {schema_name}: {e}\nRaw:\n{raw}") from e


# =========================
# 5) Utilities: sentence parsing/validation + logging
# =========================

SENT_LINE_RE = re.compile(r"^\s*(\d+)\s+(.*)\s*$")


def parse_sentences(sentences_blob: str) -> Tuple[List[int], Dict[int, str]]:
    """
    Parse the dataset's 'Sentences' field which looks like:
      "0 sentence...\n1 sentence...\n2 sentence..."
    Returns:
      - ordered_ids: [0,1,2,...]
      - id_to_sentence: {0:"...", 1:"...", ...}
    """
    ordered_ids: List[int] = []
    id_to_sentence: Dict[int, str] = {}
    for line in sentences_blob.splitlines():
        line = line.strip()
        if not line:
            continue
        m = SENT_LINE_RE.match(line)
        if not m:
            # If the dataset has wrapped lines (like "98/62 mm" then "Hg."),
            # it may break strict line parsing. In that case, we keep the line
            # but do not treat it as a new sentence ID.
            # You can improve this by pre-joining wrapped lines in preprocessing.
            continue
        sid = int(m.group(1))
        sent = m.group(2).strip()
        ordered_ids.append(sid)
        id_to_sentence[sid] = sent
    return ordered_ids, id_to_sentence


def validate_sentence_id(valid_ids: List[int], sid: int, error_flag: int) -> Tuple[bool, str]:
    if error_flag == 0:
        if sid != -1:
            return False, "error_flag_0_requires_sentence_id_-1"
        return True, ""
    # error_flag == 1
    if sid not in valid_ids:
        return False, "invalid_error_sentence_id"
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


def escape_submission_field(s: str) -> str:
    """
    Submission example shows corrected sentence quoted. Safest is:
      - If NA: return NA (no quotes)
      - Else: wrap in double quotes and escape any existing quotes.
    """
    if s == "NA":
        return "NA"
    return '"' + s.replace('"', '\\"') + '"'


# =========================
# 6) Orchestrator
# =========================

class Orchestrator:
    def __init__(self, cfg: LLMConfig, log_path: str = "logs/med_error_runs.jsonl"):
        self.cfg = cfg
        self.logger = JSONLLogger(log_path)

    def run_one(self, text_id: str, text: str, sentences_blob: str) -> FinalOut:
        run_id = str(uuid.uuid4())
        ordered_ids, id_to_sentence = parse_sentences(sentences_blob)

        self.logger.log(
            {
                "run_id": run_id,
                "step": "input",
                "text_id": text_id,
                "text": text,
                "sentences_blob": sentences_blob,
                "parsed_sentence_ids": ordered_ids,
            }
        )

        # A) Detect
        det: DetectorOut = parse_strict(
            DetectorOut,
            DETECTOR_SYSTEM,
            f"Text ID: {text_id}\n\nClinical text:\n{text}\n",
            self.cfg,
            schema_name="detector_out",
            schema_dict=DETECTOR_SCHEMA,
        )  # type: ignore
        self.logger.log({"run_id": run_id, "step": "detector", "out": det.model_dump()})

        # If confidently no error, finalize immediately
        if det.error_flag == 0 and det.certainty >= 0.85:
            final = FinalOut(
                text_id=text_id,
                error_flag=0,
                error_sentence_id=-1,
                corrected_sentence="NA",
                certainty=det.certainty,
                needs_human_review=False,
            )
            self.logger.log({"run_id": run_id, "step": "final", "out": final.model_dump()})
            return final

        # B) Localize sentence ID (uses the provided sentence list)
        localizer_user = (
            f"Text ID: {text_id}\n\n"
            f"Clinical text:\n{text}\n\n"
            f"Sentences (ID + sentence):\n{sentences_blob}\n\n"
            f"Detector output:\n{det.model_dump_json()}\n\n"
            f"Return the error_sentence_id."
        )
        print(localizer_user)
        loc: LocalizerOut = parse_strict(
            LocalizerOut,
            LOCALIZER_SYSTEM,
            localizer_user,
            self.cfg,
            schema_name="localizer_out",
            schema_dict=LOCALIZER_SCHEMA,
        )  # type: ignore
        print(loc.model_dump())
        self.logger.log({"run_id": run_id, "step": "localizer", "out": loc.model_dump()})

        ok, reason = validate_sentence_id(ordered_ids, loc.error_sentence_id, det.error_flag)
        issues: List[str] = []
        if not ok:
            issues.append(reason)

        # If error_flag=0, enforce -1 and NA
        if det.error_flag == 0:
            final = FinalOut(
                text_id=text_id,
                error_flag=0,
                error_sentence_id=-1,
                corrected_sentence="NA",
                certainty=min(det.certainty, loc.certainty),
                needs_human_review=False if det.certainty >= 0.6 else True,
            )
            self.logger.log({"run_id": run_id, "step": "final", "out": final.model_dump(), "issues": issues})
            return final

        # For error_flag=1, if sentence ID is invalid, route to human review
        if det.error_flag == 1 and (not ok):
            final = FinalOut(
                text_id=text_id,
                error_flag=1,
                error_sentence_id=-1,
                corrected_sentence="NA",
                certainty=min(det.certainty, loc.certainty, 0.5),
                needs_human_review=True,
            )
            self.logger.log({"run_id": run_id, "step": "final", "out": final.model_dump(), "issues": issues})
            return final

        # C) Correct only that sentence
        erroneous_sentence = id_to_sentence.get(loc.error_sentence_id, "")
        corrector_user = (
            f"Text ID: {text_id}\n\n"
            f"Clinical text:\n{text}\n\n"
            f"Sentences (ID + sentence):\n{sentences_blob}\n\n"
            f"Error sentence ID: {loc.error_sentence_id}\n"
            f"Error sentence text: {erroneous_sentence}\n\n"
            f"Generate the corrected version of ONLY this sentence."
        )
        cor: CorrectorOut = parse_strict(
            CorrectorOut,
            CORRECTOR_SYSTEM,
            corrector_user,
            self.cfg,
            schema_name="corrector_out",
            schema_dict=CORRECTOR_SCHEMA,
        )  # type: ignore
        self.logger.log({"run_id": run_id, "step": "corrector", "out": cor.model_dump(), "error_sentence_id": loc.error_sentence_id})

        # D) Reviewer/Judge produces final submission fields
        reviewer_user = (
            f"Text ID: {text_id}\n\n"
            f"Clinical text:\n{text}\n\n"
            f"Sentences (ID + sentence):\n{sentences_blob}\n\n"
            f"Detector:\n{det.model_dump_json()}\n\n"
            f"Localizer:\n{loc.model_dump_json()}\n\n"
            f"Corrector:\n{cor.model_dump_json()}\n\n"
            f"Task rules:\n"
            f"- If error_flag=0: error_sentence_id=-1 and corrected_sentence='NA'\n"
            f"- If error_flag=1: error_sentence_id must be a valid ID and corrected_sentence must be the corrected version of that sentence.\n"
            f"Return the final fields."
        )
        rev: ReviewerOut = parse_strict(
            ReviewerOut,
            REVIEWER_SYSTEM,
            reviewer_user,
            self.cfg,
            schema_name="reviewer_out",
            schema_dict=REVIEWER_SCHEMA,
        )  # type: ignore
        self.logger.log({"run_id": run_id, "step": "reviewer", "out": rev.model_dump()})

        # Final local validation
        final_issues: List[str] = list(rev.issues)
        ok2, reason2 = validate_sentence_id(ordered_ids, rev.final.error_sentence_id, rev.final.error_flag)
        if not ok2:
            final_issues.append(reason2)
            rev.final.needs_human_review = True
            rev.decision = "HUMAN_REVIEW"

        if rev.final.error_flag == 0:
            if rev.final.corrected_sentence != "NA" or rev.final.error_sentence_id != -1:
                final_issues.append("no_error_must_output_-1_and_NA")
                rev.final.corrected_sentence = "NA"
                rev.final.error_sentence_id = -1
                rev.final.needs_human_review = True
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

    def to_submission_line(self, final: FinalOut) -> str:
        """
        Convert FinalOut to the required submission line format:
          [Text ID] [Error Flag] [Error sentence ID or -1] [Corrected sentence or NA]
        """
        return f"{final.text_id} {final.error_flag} {final.error_sentence_id} {escape_submission_field(final.corrected_sentence)}"


# =========================
# 7) Example usage (single instance)
# =========================

def load_indices(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))["indices"]

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/med_error_run_{timestamp}.jsonl"
    model = "gpt-4.1"
    cfg = LLMConfig(model=model, temperature=0.0, max_output_tokens=600)
    orch = Orchestrator(cfg, log_path=log_file)

    split = "test"
    dataset_test = load_dataset("mkieffer/MEDEC", split=split)

    loaded_indices = load_indices("/home/heewon/workspaces/courses/cs263nlp-26w/CS263-MEDIQA-CORR-2024/baseline-experiment/sampled_test_indices.json")
    subset_test = dataset_test.select(loaded_indices)
    num_data = len(subset_test)
    
    # # Example based on your ms-train-0 style
    # text_id = "ms-train-0"
    # text = (
    #     "A 53-year-old man comes to the physician because of a 1-day history of fever and chills, "
    #     "severe malaise, and cough with yellow-green sputum. He works as a commercial fisherman on Lake Superior. "
    #     "Current medications include metoprolol and warfarin. His temperature is 38.5 C (101.3 F), pulse is 96/min, "
    #     "respirations are 26/min, and blood pressure is 98/62 mm Hg. Examination shows increased fremitus and bronchial "
    #     "breath sounds over the right middle lung field. After reviewing imaging, the causal pathogen was determined to be "
    #     "Haemophilus influenzae. An x-ray of the chest showed consolidation of the right upper lobe."
    # )
    # sentences_blob = (
    #     "0 A 53-year-old man comes to the physician because of a 1-day history of fever and chills, severe malaise, and cough with yellow-green sputum.\n"
    #     "1 He works as a commercial fisherman on Lake Superior.\n"
    #     "2 Current medications include metoprolol and warfarin.\n"
    #     "3 His temperature is 38.5 C (101.3 F), pulse is 96/min, respirations are 26/min, and blood pressure is 98/62 mm Hg.\n"
    #     "4 Examination shows increased fremitus and bronchial breath sounds over the right middle lung field.\n"
    #     "5 After reviewing imaging, the causal pathogen was determined to be Haemophilus influenzae.\n"
    #     "6 An x-ray of the chest showed consolidation of the right upper lobe."
    # )

    # final = orch.run_one(text_id=text_id, text=text, sentences_blob=sentences_blob)
    # print(final.model_dump_json(indent=2, ensure_ascii=False))
    # print("\nSubmission line:")
    # print(orch.to_submission_line(final))

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"medec-ms_{orch.cfg.model}_multi-agent_{split}_{timestamp}.txt"

    with out_path.open("w", encoding="utf-8") as f:
        for i in range(num_data):
            item = subset_test[i]
            text_id = item['text_id']
            text = item['text']
            sentences_blob = item['sentences']
            final = orch.run_one(text_id=text_id, text=text, sentences_blob=sentences_blob)
            print(final.model_dump_json(indent=2, ensure_ascii=False))
            #print("\nSubmission line:")
            out_text = orch.to_submission_line(final)
            print(out_text)
            f.write(out_text + "\n")

