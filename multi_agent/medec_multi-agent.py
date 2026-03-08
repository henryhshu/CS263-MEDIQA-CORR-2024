import os
import json
import argparse
from typing import List, Optional, Literal, Dict, Any

from datasets import load_dataset
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm


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
# 1) Pydantic Schemas
# =========================
ErrorType = Literal["diagnosis", "management", "treatment", "pharmacotherapy", "causalOrganism"]
ErrorTypeOrUnknown = Literal[
    "diagnosis", "management", "treatment", "pharmacotherapy", "causalOrganism", "unknown", "NA"
]
RiskLevel = Literal["low", "medium", "high"]


class SuspectedType(BaseModel):
    type: ErrorType
    p: float = Field(ge=0.0, le=1.0)


class TriageOut(BaseModel):
    error_flag: bool
    suspected_types: List[SuspectedType] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class Candidate(BaseModel):
    sentence_id: int
    score: float = Field(ge=0.0, le=1.0)
    type: ErrorTypeOrUnknown
    why: str
    conflicts_with: List[int] = Field(default_factory=list)


class LocatorOut(BaseModel):
    candidates: List[Candidate] = Field(default_factory=list)


class EditItem(BaseModel):
    before: str
    after: str
    change_type: Literal[
        "entity_swap", "dosage", "route", "frequency", "negation", "temporal", "other"
    ]


class Proposal(BaseModel):
    rank: int
    sentence_id: int
    corrected_sentence: str
    edit_summary: List[EditItem] = Field(default_factory=list)
    justification: str
    confidence: float = Field(ge=0.0, le=1.0)


class EditorOut(BaseModel):
    n_best: List[Proposal] = Field(default_factory=list)


class Verdict(BaseModel):
    rank: int
    accept: bool
    risk_level: RiskLevel
    issues: List[str] = Field(default_factory=list)


class CriticOut(BaseModel):
    verdicts: List[Verdict] = Field(default_factory=list)
    overall_recommendation: Literal["choose_rank_1", "choose_rank_2", "choose_rank_3", "abstain"]


class FinalOut(BaseModel):
    text_id: str
    error_flag: bool
    error_type: ErrorTypeOrUnknown
    error_sentence_id: Optional[int] = None
    corrected_sentence: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel
    edit_summary: List[EditItem] = Field(default_factory=list)


# =========================
# 2) Prompts
# =========================
COMMON_INSTRUCTIONS = """You are a medical-note consistency auditor for a benchmark task.
You must NOT provide clinical advice. Your job is only to detect inconsistencies and propose minimal text corrections.

Rules:
- Use ONLY the content in the provided note.
- Do NOT invent new symptoms, labs, diagnoses, medications, or procedures.
- Prefer minimal edits: change as little as possible to resolve the inconsistency.
- Sentence IDs are integers shown as "<id> | <sentence>".
- If uncertain, abstain with low confidence.
- Output must match the provided JSON schema (no extra keys, no prose).
"""


def triage_prompt(sentences_block: str) -> str:
    return f"""TASK: Decide whether the note contains a medical error (0/1). If yes, estimate likely error types.
ERROR TYPES: diagnosis, management, treatment, pharmacotherapy, causalOrganism.

Return JSON strictly matching schema.

SENTENCES:
{sentences_block}
"""


def locator_prompt(sentences_block: str, triage: TriageOut, k: int = 3) -> str:
    return f"""TASK: Identify up to K={k} sentence IDs that most likely contain the error.
For each candidate, cite internal contradictions by referencing other sentence IDs.

Return JSON strictly matching schema.

TRIAGE:
{triage.model_dump_json(indent=2)}

SENTENCES:
{sentences_block}
"""


def editor_prompt(
    sentences_block: str,
    target_id: int,
    target_sentence: str,
    conflicts_with: List[int],
    n: int = 3,
) -> str:
    return f"""TASK: Rewrite ONLY the target sentence to fix the error with minimal changes.
Generate N={n} alternatives (n-best). Do not modify any other sentences.

Return JSON strictly matching schema.

SENTENCES:
{sentences_block}

TARGET:
- sentence_id: {target_id}
- original_sentence: {target_sentence}

EVIDENCE:
- conflicts_with_sentence_ids: {conflicts_with}
"""


def critic_prompt(sentences_block: str, editor_out: EditorOut) -> str:
    return f"""TASK: Evaluate each proposed correction for:
(1) internal consistency with the note,
(2) introducing new unsupported medical facts,
(3) risky changes without support.

Pick the best rank or abstain.

Return JSON strictly matching schema.

SENTENCES:
{sentences_block}

PROPOSALS:
{editor_out.model_dump_json(indent=2)}
"""


def arbiter_prompt(
    text_id: str,
    sentences_block: str,
    triage: TriageOut,
    locator: LocatorOut,
    editor_out: EditorOut,
    critic_out: CriticOut,
) -> str:
    return f"""TASK: Produce the FINAL output fields for the benchmark:
- error_flag (boolean)
- error_type (one of: diagnosis, management, treatment, pharmacotherapy, causalOrganism, unknown, NA)
- error_sentence_id (integer id of the erroneous sentence, or null if none)
- corrected_sentence (string rewrite of that sentence, or null if none)

Rules:
- If triage says no error, set:
  - error_flag=false, error_type="NA", error_sentence_id=null, corrected_sentence=null
- If critic says abstain, you may still set error_flag true if evidence is strong, but then:
  - error_type="unknown"
  - error_sentence_id may be null
  - corrected_sentence may be null
  - confidence low
- Do NOT modify any sentence other than the chosen error_sentence_id.
- corrected_sentence must be a single sentence (no extra commentary).

Return JSON strictly matching schema.

text_id: {text_id}

TRIAGE:
{triage.model_dump_json(indent=2)}

LOCATOR:
{locator.model_dump_json(indent=2)}

EDITOR:
{editor_out.model_dump_json(indent=2)}

CRITIC:
{critic_out.model_dump_json(indent=2)}

SENTENCES:
{sentences_block}
"""


# =========================
# 3) Helpers
# =========================
def parse_sentences_block(sentences_field: str) -> List[Dict[str, Any]]:
    """
    Input example line: "5 | After reviewing imaging, ..."
    """
    out = []
    for line in sentences_field.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        left, right = line.split("|", 1)
        sid = int(left.strip())
        sent = right.strip()
        out.append({"id": sid, "sentence": sent})
    return out


def to_sentences_block(sents: List[Dict[str, Any]]) -> str:
    return "\n".join([f'{x["id"]} | {x["sentence"]}' for x in sents])


def call_parse(client: OpenAI, model: str, instructions: str, user_input: str, schema):
    """
    Uses Responses API structured outputs via pydantic schema.
    NOTE: Do not pass temperature unless you know the model supports it.
    """
    resp = client.responses.parse(
        model=model,
        instructions=instructions,
        input=user_input,
        text_format=schema,
    )
    parsed = resp.output_parsed
    if parsed is None:
        return schema.model_validate_json(resp.output_text)
    return parsed


def pick_best_candidate(locator: LocatorOut, valid_ids: set) -> Optional[Candidate]:
    cands = [c for c in locator.candidates if c.sentence_id in valid_ids]
    cands.sort(key=lambda c: c.score, reverse=True)
    return cands[0] if cands else None


def pick_best_correction(editor: EditorOut, critic: CriticOut) -> Optional[Proposal]:
    if not editor.n_best:
        return None
    # critic recommendation
    rec = critic.overall_recommendation
    if rec.startswith("choose_rank_"):
        try:
            r = int(rec.split("_")[-1])
            for p in editor.n_best:
                if p.rank == r and p.corrected_sentence.strip():
                    return p
        except Exception:
            pass
    # else pick first non-empty
    for p in sorted(editor.n_best, key=lambda x: x.rank):
        if p.corrected_sentence.strip():
            return p
    return None


def to_submission_line(text_id: str, error_flag: bool, sid: int, corrected: str) -> str:
    """
    Required output:
    text-id 0 -1 NA
    text-id 1 8 "..."
    """
    if not error_flag:
        return f"{text_id} 0 -1 NA"
    # error case
    if sid is None or sid < 0:
        sid = -1
    if not corrected or corrected == "NA":
        # unavoidable fallback (should rarely happen)
        return f"{text_id} 1 {sid} NA"
    # Always quote corrected sentence safely using JSON encoding
    return f"{text_id} 1 {sid} {json.dumps(corrected, ensure_ascii=False)}"


# =========================
# 4) Multi-agent Pipeline
# =========================
class PipelineConfig(BaseModel):
    triage_model: str = "gpt-4o-mini"
    locator_model: str = "gpt-4o-mini"
    editor_model: str = "gpt-5.2"
    critic_model: str = "gpt-5.2"
    arbiter_model: str = "gpt-5.2"
    triage_threshold: float = 0.35
    k_candidates: int = 3
    n_best: int = 3


def run_one(client: OpenAI, row: Dict[str, Any], cfg: PipelineConfig) -> str:
    text_id = row["text_id"]
    sents = parse_sentences_block(row["sentences"])
    sentences_block = to_sentences_block(sents)
    valid_ids = {x["id"] for x in sents}
    id2sent = {x["id"]: x["sentence"] for x in sents}

    # 1) TRIAGE
    triage = call_parse(
        client,
        cfg.triage_model,
        COMMON_INSTRUCTIONS,
        triage_prompt(sentences_block),
        TriageOut,
    )
    max_p = max([st.p for st in triage.suspected_types], default=0.0)

    if (not triage.error_flag) or (max_p < cfg.triage_threshold):
        return to_submission_line(text_id, False, -1, "NA")

    # 2) LOCATOR
    locator = call_parse(
        client,
        cfg.locator_model,
        COMMON_INSTRUCTIONS,
        locator_prompt(sentences_block, triage, k=cfg.k_candidates),
        LocatorOut,
    )
    best = pick_best_candidate(locator, valid_ids)
    if best is None:
        # can’t localize -> fallback: mark as no error (or keep error with NA)
        # Here we keep error=1 but unknown localization (submission will be penalized but avoids crash)
        return to_submission_line(text_id, True, -1, "NA")

    target_id = best.sentence_id
    target_sentence = id2sent[target_id]

    # 3) EDITOR
    editor_out = call_parse(
        client,
        cfg.editor_model,
        COMMON_INSTRUCTIONS,
        editor_prompt(
            sentences_block,
            target_id=target_id,
            target_sentence=target_sentence,
            conflicts_with=best.conflicts_with,
            n=cfg.n_best,
        ),
        EditorOut,
    )

    # 4) CRITIC
    critic_out = call_parse(
        client,
        cfg.critic_model,
        COMMON_INSTRUCTIONS,
        critic_prompt(sentences_block, editor_out),
        CriticOut,
    )

    # 5) ARBITER
    final = call_parse(
        client,
        cfg.arbiter_model,
        COMMON_INSTRUCTIONS,
        arbiter_prompt(text_id, sentences_block, triage, locator, editor_out, critic_out),
        FinalOut,
    )

    # 6) Build submission fields with robust fallbacks
    if not final.error_flag:
        return to_submission_line(text_id, False, -1, "NA")

    sid = final.error_sentence_id if final.error_sentence_id in valid_ids else target_id
    corrected = (final.corrected_sentence or "").strip()

    if not corrected:
        best_prop = pick_best_correction(editor_out, critic_out)
        corrected = best_prop.corrected_sentence.strip() if best_prop else ""

    if not corrected:
        # last resort
        corrected = "NA"

    return to_submission_line(text_id, True, sid, corrected)

def load_indices(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))["indices"]

# =========================
# 5) Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation", "test"])
    parser.add_argument("--limit", type=int, default=0, help="0 means all")
    parser.add_argument("--out", type=str, default=None, help="submission output txt path")
    parser.add_argument("--triage_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--locator_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--editor_model", type=str, default="gpt-5.2")
    parser.add_argument("--critic_model", type=str, default="gpt-5.2")
    parser.add_argument("--arbiter_model", type=str, default="gpt-5.2")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Please set OPENAI_API_KEY env var.")

    client = OpenAI()

    ds = load_dataset("mkieffer/MEDEC", split=args.split)
    # if args.limit and args.limit > 0:
    #     ds = ds.select(range(min(args.limit, len(ds))))

    loaded_indices = load_indices("/home/heewon/workspaces/cs263nlp/CS263-MEDIQA-CORR-2024/baseline-experiment/sampled_test_indices.json")
    ds = ds.select(loaded_indices)
    num_data = len(ds)
    
    cfg = PipelineConfig(
        triage_model=args.triage_model,
        locator_model=args.locator_model,
        editor_model=args.editor_model,
        critic_model=args.critic_model,
        arbiter_model=args.arbiter_model,
    )

    if args.out is None:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = f"medec_multi-agent_submission_{args.split}_{ts}.txt"

    with open(args.out, "w", encoding="utf-8") as f:
        for row in tqdm(ds, desc=f"MEDEC {args.split} -> submission"):
            try:
                line = run_one(client, row, cfg)
            except Exception:
                # Fail-safe: output "no error" to keep format valid
                line = to_submission_line(row.get("text_id", "UNKNOWN"), False, -1, "NA")
            f.write(line + "\n")

    print(f"[OK] Saved submission file: {args.out}")


if __name__ == "__main__":
    main()