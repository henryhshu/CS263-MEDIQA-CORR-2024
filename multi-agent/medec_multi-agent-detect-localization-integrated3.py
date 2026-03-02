import os
import json
import argparse
from typing import List, Literal, Dict, Any

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
# 1) Schemas (Structured Outputs)
# =========================
Verdict = Literal["CORRECT", "ERROR"]
RiskLevel = Literal["low", "medium", "high"]


class DetectorOut(BaseModel):
    verdict: Verdict
    error_sentence_id: int  # -1 if CORRECT
    corrected_sentence: str  # "NA" if CORRECT
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str  # short


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


class VerdictItem(BaseModel):
    rank: int
    accept: bool
    risk_level: RiskLevel
    issues: List[str] = Field(default_factory=list)


class CriticOut(BaseModel):
    verdicts: List[VerdictItem] = Field(default_factory=list)
    overall_recommendation: Literal[
        "choose_rank_1", "choose_rank_2", "choose_rank_3", "choose_rank_4", "choose_rank_5"
    ]

# =========================
# 2) Prompts
# =========================
DETECTOR_INSTRUCTIONS = """You are a skilled medical doctor reviewing clinical text for ONE possible medical error.
You must NOT provide clinical advice. This is a benchmark task only.

The text has one sentence per line.
Each line starts with the sentence ID, followed by a pipe character, then the sentence.

The text is either correct or contains one error related to treatment, management, cause, or diagnosis.
Check every sentence.

If the text is correct, output CORRECT.
If the text has an error, output the sentence id containing the error and a corrected version of that sentence.

IMPORTANT OUTPUT RULE:
Return STRICT JSON that matches the provided schema exactly. No extra keys. No prose outside JSON.
If correct: verdict="CORRECT", error_sentence_id=-1, corrected_sentence="NA".
If error: verdict="ERROR", error_sentence_id=<id>, corrected_sentence=<single corrected sentence>.
"""

# DETECTOR_INSTRUCTIONS = """You are a skilled medical doctor reviewing clinical text for ONE possible medical error.
# You must NOT provide clinical advice. This is a benchmark task only.

# The text has one sentence per line.
# Each line starts with the sentence ID, followed by a pipe character, then the sentence.

# The text is either correct or contains one error related to treatment, management, cause, or diagnosis.
# Check every sentence.

# Error type definitions:
# - Diagnosis error: an incorrect, inconsistent, or unsupported diagnosis/assessment stated in the text.
# - Management error: an incorrect or inappropriate clinical plan, disposition, follow-up, monitoring, or non-procedural care decision.
# - Treatment error: an incorrect or inappropriate therapeutic intervention or procedure (excluding medication-specific details).
# - Pharmacotherapy error: an incorrect medication choice or medication details such as dose, route, frequency, duration, or drug interactions/contraindications.
# - Causal organism error: an incorrect infectious agent or etiology stated as the cause of the condition when it conflicts with the clinical evidence in the note.

# If the text is correct, output CORRECT.
# If the text has an error, output the sentence id containing the error and a corrected version of that sentence.

# IMPORTANT OUTPUT RULE:
# Return STRICT JSON that matches the provided schema exactly. No extra keys. No prose outside JSON.
# If correct: verdict="CORRECT", error_sentence_id=-1, corrected_sentence="NA".
# If error: verdict="ERROR", error_sentence_id=<id>, corrected_sentence=<single corrected sentence>.
# """

# DETECTOR_INSTRUCTIONS = """You are a skilled medical doctor reviewing clinical text for ONE possible medical error.
# You must NOT provide clinical advice. This is a benchmark task only.

# The text has one sentence per line.
# Each line starts with the sentence ID, followed by a pipe character, then the sentence.

# The text is either correct or contains one error related to treatment, management, cause, or diagnosis.
# Check every sentence.

# Error type definitions:
# - Diagnosis - The provided diagnosis is inaccurate.
# - Management - The next step provided in management is inaccurate.
# - Pharmacotherapy - The recommended pharmacotherapy is inaccurate.
# - Treatment - The recommended treatment is inaccurate.
# - Causal Organism - The indicated causal organism or causal pathogen is inaccurate.

# If the text is correct, output CORRECT.
# If the text has an error, output the sentence id containing the error and a corrected version of that sentence.

# IMPORTANT OUTPUT RULE:
# Return STRICT JSON that matches the provided schema exactly. No extra keys. No prose outside JSON.
# If correct: verdict="CORRECT", error_sentence_id=-1, corrected_sentence="NA".
# If error: verdict="ERROR", error_sentence_id=<id>, corrected_sentence=<single corrected sentence>.
# """

# CRITIC_INSTRUCTIONS = """You are a clinical-text consistency critic for a benchmark.
# You must NOT provide clinical advice.

# Evaluate proposed correction(s) for:
# (1) internal consistency with the note,
# (2) introducing new unsupported medical facts,
# (3) risky changes (new drug/dose/diagnosis) without support,
# (4) not being a single sentence.

# Return STRICT JSON matching schema exactly.
# """

CRITIC_INSTRUCTIONS = """You are a clinical-text consistency critic for a benchmark.
You must NOT provide clinical advice.

You will evaluate proposed correction(s) for a medical narrative that is either correct or contains ONE medical error.
Each line in the narrative starts with a sentence ID, followed by a pipe character, then the sentence.

Possible error types (for evaluation):
- Diagnosis - The provided diagnosis is inaccurate.
- Management - The next step provided in management is inaccurate.
- Pharmacotherapy - The recommended pharmacotherapy is inaccurate.
- Treatment - The recommended treatment is inaccurate.
- Causal Organism - The indicated causal organism or causal pathogen is inaccurate.

Evaluation criteria for each proposed correction:
1) Internal consistency: The correction must not conflict with other sentences in the note.
2) No new unsupported facts: Do NOT add new symptoms, labs, diagnoses, medications, procedures, organisms, or timelines not supported by the note.
3) Minimal edit: Prefer the smallest change that fixes the error.
4) Risk control: Avoid risky new clinical actions (e.g., new drug/dose/diagnosis) without support in the note.
5) Wrong-type fix: Reject proposals that fix the wrong type or change unrelated clinical content.
6) Single-sentence constraint: The corrected_sentence must be a single sentence (no extra commentary, no multiple sentences).

CRITICAL DECISION RULE:
You MUST choose exactly one proposal rank as the final selection.
Do NOT abstain. Even if all proposals have issues, pick the least-bad option.
Set accept=false and include issues for rejected proposals, but still choose one rank in overall_recommendation.

IMPORTANT OUTPUT RULE:
Return STRICT JSON that matches the provided schema exactly. No extra keys. No prose outside JSON.
"""

EDITOR_INSTRUCTIONS = """You are an editor for a benchmark clinical note correction task.
You must NOT provide clinical advice.

Rewrite ONLY the target sentence to fix the error with minimal changes.
Do not modify any other sentences.
Return STRICT JSON matching schema exactly.
"""


def detector_prompt(sentences_block: str) -> str:
    return f"""SENTENCES:
{sentences_block}
"""


def critic_prompt(sentences_block: str, editor_out: EditorOut) -> str:
    return f"""SENTENCES:
{sentences_block}

PROPOSALS:
{editor_out.model_dump_json(indent=2)}
"""


def editor_prompt(sentences_block: str, target_id: int, target_sentence: str, n: int = 3) -> str:
    return f"""TASK: Rewrite ONLY the target sentence to fix the error with minimal changes.
Generate N={n} alternatives (n-best). Do not modify any other sentences.

SENTENCES:
{sentences_block}

TARGET:
- sentence_id: {target_id}
- original_sentence: {target_sentence}
"""


# =========================
# 3) Helpers
# =========================
def parse_sentences_field(sentences_field: str) -> List[Dict[str, Any]]:
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
    # Structured Outputs via responses.parse
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


def to_submission_line(text_id: str, error_flag: int, sid: int, corrected: str) -> str:
    if error_flag == 0:
        return f"{text_id} 0 -1 NA"
    # always quote corrected sentence safely
    if not corrected or corrected == "NA":
        return f"{text_id} 1 {sid} NA"
    return f"{text_id} 1 {sid} {json.dumps(corrected, ensure_ascii=False)}"


def make_single_proposal(sentence_id: int, corrected_sentence: str) -> EditorOut:
    return EditorOut(
        n_best=[
            Proposal(
                rank=1,
                sentence_id=sentence_id,
                corrected_sentence=corrected_sentence,
                edit_summary=[],
                justification="From detector integrated output.",
                confidence=0.5,
            )
        ]
    )


def pick_recommended(editor_out: EditorOut, critic_out: CriticOut) -> str:
    rec = critic_out.overall_recommendation
    if rec.startswith("choose_rank_"):
        r = int(rec.split("_")[-1])
        for p in editor_out.n_best:
            if p.rank == r:
                return p.corrected_sentence.strip()
    # fallback: first non-empty
    for p in sorted(editor_out.n_best, key=lambda x: x.rank):
        if p.corrected_sentence.strip():
            return p.corrected_sentence.strip()
    return ""


# =========================
# 4) Integrated Pipeline
# =========================
class Config(BaseModel):
    detector_model: str = "gpt-5.2"
    critic_model: str = "gpt-5.2"
    editor_model: str = "gpt-5.2"
    n_best: int = 5


def run_one(client: OpenAI, row: Dict[str, Any], cfg: Config) -> str:
    text_id = row["text_id"]
    sents = parse_sentences_field(row["sentences"])
    sentences_block = to_sentences_block(sents)
    valid_ids = {x["id"] for x in sents}
    id2sent = {x["id"]: x["sentence"] for x in sents}

    # (1) Detector = Triage + Localization (+ correction) in one shot
    det: DetectorOut = call_parse(
        client,
        cfg.detector_model,
        DETECTOR_INSTRUCTIONS,
        detector_prompt(sentences_block),
        DetectorOut,
    )

    if det.verdict == "CORRECT" or det.error_sentence_id == -1:
        return to_submission_line(text_id, 0, -1, "NA")

    sid = det.error_sentence_id
    if sid not in valid_ids:
        # invalid localization -> fail-safe
        return to_submission_line(text_id, 0, -1, "NA")

    corrected = (det.corrected_sentence or "").strip()
    if not corrected or corrected == "NA":
        # detector found error but gave no correction -> fall back to editor
        corrected = ""

    # (2) Critic checks detector’s correction first (cheap)
    if corrected:
        single = make_single_proposal(sid, corrected)
        # critic1: CriticOut = call_parse(
        #     client,
        #     cfg.critic_model,
        #     CRITIC_INSTRUCTIONS,
        #     critic_prompt(sentences_block, single),
        #     CriticOut,
        # )
        # if critic1.overall_recommendation == "choose_rank_1":
        #     return to_submission_line(text_id, 1, sid, corrected)
        try:
            critic1: CriticOut = call_parse(
                client,
                cfg.critic_model,
                CRITIC_INSTRUCTIONS,
                critic_prompt(sentences_block, single),
                CriticOut,
            )
            # single proposal only has rank=1, so choose_rank_1 means accept-as-is
            if critic1.overall_recommendation == "choose_rank_1":
                return to_submission_line(text_id, 1, sid, corrected)
        except Exception:
            # If critic fails to comply/parse, just proceed to Editor n-best stage
            pass
    
    # (3) If critic rejects (or no correction), run Editor n-best, then Critic selects
    editor_out: EditorOut = call_parse(
        client,
        cfg.editor_model,
        EDITOR_INSTRUCTIONS,
        editor_prompt(sentences_block, sid, id2sent[sid], n=cfg.n_best),
        EditorOut,
    )
    # critic2: CriticOut = call_parse(
    #     client,
    #     cfg.critic_model,
    #     CRITIC_INSTRUCTIONS,
    #     critic_prompt(sentences_block, editor_out),
    #     CriticOut,
    # )
    # final_corrected = pick_recommended(editor_out, critic2)
    try:
        critic2: CriticOut = call_parse(
            client,
            cfg.critic_model,
            CRITIC_INSTRUCTIONS,
            critic_prompt(sentences_block, editor_out),
            CriticOut,
        )
        final_corrected = pick_recommended(editor_out, critic2)
    except Exception:
        # Fail-safe: pick rank 1 if critic output cannot be parsed
        final_corrected = editor_out.n_best[0].corrected_sentence.strip() if editor_out.n_best else ""

    if not final_corrected:
        final_corrected = "NA"
    if not final_corrected:
        final_corrected = "NA"

    return to_submission_line(text_id, 1, sid, final_corrected)

def load_indices(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))["indices"]

# =========================
# 5) Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation", "test"])
    parser.add_argument("--limit", type=int, default=0, help="0 means all")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--detector_model", type=str, default="gpt-5.2")
    parser.add_argument("--critic_model", type=str, default="gpt-5.2")
    parser.add_argument("--editor_model", type=str, default="gpt-5.2")
    parser.add_argument("--n_best", type=int, default=3)
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
    main()