#!/usr/bin/env python3
"""
Multi-agent + In-Context Learning runner for MEDEC.

This file reuses:
- multi_agent_detect_critic_edit.py
- in_context_learning.py

It keeps the multi-agent pipeline (Detector -> Critic -> Editor -> Critic),
but augments prompts with fixed-shot or retrieval-based in-context examples.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Literal

from openai import OpenAI
from pydantic import BaseModel
from datasets import load_dataset

# Reuse the existing multi-agent pipeline components
import multi_agent.multi_agent_detect_critic_edit as ma

# Reuse the existing ICL / retrieval utilities
import in_context_learning.in_context_learning as icl


# =========================
# 1) Config
# =========================
ICLMode = Literal["none", "fixed", "dynamic"]


class ICLConfig(BaseModel):
    mode: ICLMode = "dynamic"
    k_shot: int = 3
    embedding_model: str = icl.DEFAULT_EMBEDDING_MODEL
    cache_dir: Path = Path("in-context-learning/cache")
    use_detector_icl: bool = True
    use_editor_icl: bool = True
    use_critic_icl: bool = False


class RunConfig(BaseModel):
    detector_model: str = "gpt-4.1"
    critic_model: str = "gpt-4.1"
    editor_model: str = "gpt-4.1"
    n_best: int = 3
    temperature: float = 0.0
    reasoning_effort: str | None = None

# =========================
# 2) Example builders
# =========================
def _gold_detector_output(item: dict[str, Any]) -> ma.DetectorOut:
    if item["error_flag"] == 0:
        return ma.DetectorOut(
            verdict="CORRECT",
            error_sentence_id=-1,
            corrected_sentence="NA",
            confidence=1.0,
            rationale="The note is internally consistent and contains no benchmark error.",
        )
    return ma.DetectorOut(
        verdict="ERROR",
        error_sentence_id=int(item["error_sentence_id"]),
        corrected_sentence=str(item["corrected_sentence"]),
        confidence=1.0,
        rationale="The note contains one benchmark medical error; the corrected sentence fixes it minimally.",
    )


def _gold_editor_output(item: dict[str, Any]) -> ma.EditorOut:
    if item["error_flag"] == 0:
        return ma.EditorOut(n_best=[])
    return ma.EditorOut(
        n_best=[
            ma.Proposal(
                rank=1,
                sentence_id=int(item["error_sentence_id"]),
                corrected_sentence=str(item["corrected_sentence"]),
                edit_summary=[],
                justification="Minimal edit that matches the gold correction.",
                confidence=1.0,
            )
        ]
    )


def _gold_critic_output(item: dict[str, Any]) -> ma.CriticOut:
    if item["error_flag"] == 0:
        return ma.CriticOut(
            verdicts=[],
            overall_recommendation="abstain",
        )
    return ma.CriticOut(
        verdicts=[
            ma.VerdictItem(
                rank=1,
                accept=True,
                risk_level="low",
                issues=[],
            )
        ],
        overall_recommendation="choose_rank_1",
    )


def build_detector_examples(examples: list[dict[str, Any]]) -> str:
    if not examples:
        return ""

    sections: list[str] = []
    for i, ex in enumerate(examples, start=1):
        out = _gold_detector_output(ex)
        sections.append(
            f"""[DETECTOR EXAMPLE {i}]
INPUT:
SENTENCES:
{ex["sentences"]}

OUTPUT:
{out.model_dump_json(indent=2)}
"""
        )
    return "\n".join(sections)


def build_editor_examples(examples: list[dict[str, Any]]) -> str:
    filtered = [ex for ex in examples if ex["error_flag"] == 1]
    if not filtered:
        return ""

    sections: list[str] = []
    for i, ex in enumerate(filtered, start=1):
        sents = ma.parse_sentences_field(ex["sentences"])
        id2sent = {x["id"]: x["sentence"] for x in sents}
        sid = int(ex["error_sentence_id"])
        original_sentence = id2sent.get(sid, "")
        out = _gold_editor_output(ex)

        sections.append(
            f"""[EDITOR EXAMPLE {i}]
INPUT:
TASK: Rewrite ONLY the target sentence to fix the error with minimal changes.
Generate N=1 alternatives (n-best). Do not modify any other sentences.

SENTENCES:
{ex["sentences"]}

TARGET:
- sentence_id: {sid}
- original_sentence: {original_sentence}

OUTPUT:
{out.model_dump_json(indent=2)}
"""
        )
    return "\n".join(sections)


def build_critic_examples(examples: list[dict[str, Any]]) -> str:
    filtered = [ex for ex in examples if ex["error_flag"] == 1]
    if not filtered:
        return ""

    sections: list[str] = []
    for i, ex in enumerate(filtered, start=1):
        proposal = _gold_editor_output(ex)
        out = _gold_critic_output(ex)
        sections.append(
            f"""[CRITIC EXAMPLE {i}]
INPUT:
SENTENCES:
{ex["sentences"]}

PROPOSALS:
{proposal.model_dump_json(indent=2)}

OUTPUT:
{out.model_dump_json(indent=2)}
"""
        )
    return "\n".join(sections)


# =========================
# 3) ICL example selection
# =========================
def select_examples(
    item: dict[str, Any],
    train_items: list[dict[str, Any]],
    icl_cfg: ICLConfig,
    train_embeddings: dict[str, list[float]] | None = None,
) -> list[dict[str, Any]]:
    if icl_cfg.mode == "none" or icl_cfg.k_shot <= 0:
        return []

    if icl_cfg.mode == "fixed":
        return train_items[: icl_cfg.k_shot]

    if icl_cfg.mode == "dynamic":
        if train_embeddings is None:
            raise ValueError("train_embeddings must be provided for dynamic ICL mode.")
        query_embedding = icl.cached_embedding(
            cache_dir=icl_cfg.cache_dir,
            namespace="query",
            model=icl_cfg.embedding_model,
            cache_id=item["text_id"],
            text=icl.retrieval_text(item),
        )
        return icl.nearest_examples(
            train_items=train_items,
            train_embeddings=train_embeddings,
            query_embedding=query_embedding,
            k=icl_cfg.k_shot,
        )

    raise ValueError(f"Unsupported ICL mode: {icl_cfg.mode}")


# =========================
# 4) Prompt wrappers
# =========================
def wrap_detector_prompt(sentences_block: str, examples: list[dict[str, Any]], use_icl: bool) -> str:
    base = ma.detector_prompt(sentences_block)
    if not use_icl or not examples:
        return base

    few_shot = build_detector_examples(examples)
    return f"""Use the following examples as style and task references.

{few_shot}

Now solve the next instance.

{base}
"""

def wrap_editor_prompt(
    sentences_block: str,
    target_id: int,
    target_sentence: str,
    n: int,
    examples: list[dict[str, Any]],
    use_icl: bool,
) -> str:
    base = ma.editor_prompt(sentences_block, target_id, target_sentence, n=n)
    if not use_icl or not examples:
        return base

    few_shot = build_editor_examples(examples)
    if not few_shot.strip():
        return base

    return f"""Use the following examples as style and task references.

{few_shot}

Now solve the next instance.

{base}
"""


def wrap_critic_prompt(
    sentences_block: str,
    editor_out: ma.EditorOut,
    examples: list[dict[str, Any]],
    use_icl: bool,
) -> str:
    base = ma.critic_prompt(sentences_block, editor_out)
    if not use_icl or not examples:
        return base

    few_shot = build_critic_examples(examples)
    if not few_shot.strip():
        return base

    return f"""Use the following examples as style and task references.

{few_shot}

Now solve the next instance.

{base}
"""


# =========================
# 5) LLM call wrapper
# =========================
def call_parse_with_reasoning(
    client: OpenAI,
    model: str,
    instructions: str,
    user_input: str,
    schema,
    agent_role: str,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
):
    """
    Same spirit as ma.call_parse, but optionally supports reasoning effort.
    """
    kwargs = {
        "model": model,
        "instructions": instructions,
        "input": user_input,
        "text_format": schema,
    }
    if temperature:
        kwargs["temperature"] = temperature
    if reasoning_effort:
        kwargs["reasoning"] = {"effort": reasoning_effort}

    resp = client.responses.parse(**kwargs)

    # print(f"AGENT ROLE: {agent_role}")
    # print(f"MODEL: {model}")
    # print(f"Instruction:\n{instructions}")
    # print(f"User Input:\n{user_input}")

    parsed = resp.output_parsed
    # print(parsed)

    if parsed is None:
        return schema.model_validate_json(resp.output_text)
    return parsed


# =========================
# 6) Integrated pipeline
# =========================
def run_one_with_icl(
    client: OpenAI,
    row: dict[str, Any],
    run_cfg: RunConfig,
    icl_cfg: ICLConfig,
    train_items: list[dict[str, Any]],
    train_embeddings: dict[str, list[float]] | None = None,
) -> str:
    text_id = row["text_id"]
    sents = ma.parse_sentences_field(row["sentences"])
    sentences_block = ma.to_sentences_block(sents)
    valid_ids = {x["id"] for x in sents}
    id2sent = {x["id"]: x["sentence"] for x in sents}

    examples = select_examples(
        item=row,
        train_items=train_items,
        icl_cfg=icl_cfg,
        train_embeddings=train_embeddings,
    )

    # (1) Detector
    # detector_input = wrap_detector_prompt(
    #     sentences_block=sentences_block,
    #     examples=examples,
    #     use_icl=icl_cfg.use_detector_icl,
    # )
    # print(repr(detector_input))
    # print(repr(ma.detector_prompt(sentences_block)))
    # assert detector_input == ma.detector_prompt(sentences_block)
    
    # det: ma.DetectorOut = call_parse_with_reasoning(
    #     client=client,
    #     model=run_cfg.detector_model,
    #     instructions=ma.DETECTOR_INSTRUCTIONS,
    #     user_input=detector_input,
    #     schema=ma.DetectorOut,
    #     agent_role="Detector",
    #     temperature=run_cfg.temperature,
    #     reasoning_effort=run_cfg.reasoning_effort,
    # )

    detector_input = icl.build_messages(row, examples)

    det: ma.DetectorOut = call_parse_with_reasoning(
        client=client,
        model=run_cfg.detector_model,
        instructions="",
        user_input=detector_input,
        schema=ma.DetectorOut,
        agent_role="Detector",
        temperature=run_cfg.temperature,
        reasoning_effort=run_cfg.reasoning_effort,
    )
    
    if det.verdict == "CORRECT" or det.error_sentence_id == -1:
        return ma.to_submission_line(text_id, 0, -1, "NA")

    sid = det.error_sentence_id
    if sid not in valid_ids:
        return ma.to_submission_line(text_id, 0, -1, "NA")

    corrected = (det.corrected_sentence or "").strip()
    if not corrected or corrected == "NA":
        corrected = ""

    # (2) Critic on detector proposal
    if corrected:
        single = ma.make_single_proposal(sid, corrected)
        critic1_input = wrap_critic_prompt(
            sentences_block=sentences_block,
            editor_out=single,
            examples=examples,
            use_icl=icl_cfg.use_critic_icl,
        )
        # print(repr(critic1_input))
        # print(repr(ma.critic_prompt(sentences_block, single)))
        # assert critic1_input == ma.critic_prompt(sentences_block, single)
        critic1: ma.CriticOut = call_parse_with_reasoning(
            client=client,
            model=run_cfg.critic_model,
            instructions=ma.CRITIC_INSTRUCTIONS,
            user_input=critic1_input,
            schema=ma.CriticOut,
            agent_role="Critic1",
            temperature=run_cfg.temperature,
            reasoning_effort=run_cfg.reasoning_effort,
        )
        if critic1.overall_recommendation == "choose_rank_1":
            return ma.to_submission_line(text_id, 1, sid, corrected)

    # (3) Editor n-best
    editor_input = wrap_editor_prompt(
        sentences_block=sentences_block,
        target_id=sid,
        target_sentence=id2sent[sid],
        n=run_cfg.n_best,
        examples=examples,
        use_icl=icl_cfg.use_editor_icl,
    )
    # print(repr(editor_input))
    # print(repr(ma.critic_prompt(sentences_block)))
    # assert editor_input == ma.editor_prompt(sentences_block, sid, id2sent[sid], n=run_cfg.n_best)

    editor_out: ma.EditorOut = call_parse_with_reasoning(
        client=client,
        model=run_cfg.editor_model,
        instructions=ma.EDITOR_INSTRUCTIONS,
        user_input=editor_input,
        schema=ma.EditorOut,
        agent_role="Editor",
        temperature=run_cfg.temperature,
        reasoning_effort=run_cfg.reasoning_effort,
    )

    critic2_input = wrap_critic_prompt(
        sentences_block=sentences_block,
        editor_out=editor_out,
        examples=examples,
        use_icl=icl_cfg.use_critic_icl,
    )
    # print(repr(critic2_input))
    # print(repr(ma.critic_prompt(sentences_block, editor_out)))
    # assert critic2_input == ma.critic_prompt(sentences_block, editor_out)
    critic2: ma.CriticOut = call_parse_with_reasoning(
        client=client,
        model=run_cfg.critic_model,
        instructions=ma.CRITIC_INSTRUCTIONS,
        user_input=critic2_input,
        schema=ma.CriticOut,
        agent_role="Critic2",
        temperature=run_cfg.temperature,
        reasoning_effort=run_cfg.reasoning_effort,
    )

    final_corrected = ma.pick_recommended(editor_out, critic2)
    if not final_corrected:
        final_corrected = "NA"

    return ma.to_submission_line(text_id, 1, sid, final_corrected)

def load_indices_raw(path: Path) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "indices" in payload:
        payload = payload["indices"]
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list or an object with 'indices' in {path}")
    return [int(x) for x in payload]


def load_split_raw(
    split: str,
    dataset_name: str,
    dataset_config: str | None = None,
    local_path: Path | None = None,
    indices: list[int] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """
    Load rows without canonicalize_item().
    Returns raw dataset rows as closely as possible to the original multi-agent code.
    """
    if local_path is not None:
        suffix = local_path.suffix.lower()

        if suffix == ".json":
            payload = json.loads(local_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and "data" in payload:
                payload = payload["data"]
            elif isinstance(payload, dict):
                payload = [payload]
            if not isinstance(payload, list):
                raise ValueError(f"Unsupported JSON structure in {local_path}")
            items = payload

        elif suffix == ".jsonl":
            items = []
            with local_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        items.append(json.loads(line))

        else:
            raise ValueError(f"Unsupported local file type without canonicalization: {local_path}")

        if indices is not None:
            items = [items[i] for i in indices]
        if limit is not None:
            items = items[:limit]
        return items

    # Hugging Face datasets path
    if dataset_config:
        ds = load_dataset(dataset_name, dataset_config, split=split)
    else:
        ds = load_dataset(dataset_name, split=split)

    if indices is not None:
        ds = ds.select(indices)
    elif limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    return [dict(row) for row in ds]

# =========================
# 7) CLI
# =========================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run multi-agent + ICL experiment for MEDEC.",
    )

    # dataset options (reusing in_context_learning.py style)
    parser.add_argument("--dataset-name", default=icl.DEFAULT_DATASET)
    parser.add_argument("--dataset-config", default=icl.DEFAULT_DATASET_CONFIG)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--test-file", type=Path)
    parser.add_argument("--sample-indices", type=Path, default=icl.DEFAULT_SAMPLE_INDICES)
    parser.add_argument("--use-sampled-test", dest="use_sampled_test", action="store_true")
    parser.add_argument("--no-sampled-test", dest="use_sampled_test", action="store_false")
    parser.add_argument("--full-test", action="store_true")
    parser.add_argument("--num-test-examples", type=int)
    parser.set_defaults(use_sampled_test=True)

    # multi-agent model options
    parser.add_argument("--detector-model", default="gpt-5.2")
    parser.add_argument("--critic-model", default="gpt-5.2")
    parser.add_argument("--editor-model", default="gpt-5.2")
    parser.add_argument("--n-best", type=int, default=3)
    parser.add_argument("--reasoning-effort", choices=["minimal", "low", "medium", "high"])

    # ICL options
    parser.add_argument("--icl-mode", choices=["none", "fixed", "dynamic"], default="dynamic")
    parser.add_argument("--k-shot", type=int, default=3)
    parser.add_argument("--embedding-model", default=icl.DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--cache-dir", type=Path, default=Path("in-context-learning/cache"))
    parser.add_argument("--disable-detector-icl", action="store_true")
    parser.add_argument("--disable-editor-icl", action="store_true")
    parser.add_argument("--enable-critic-icl", action="store_true")

    # output
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=Path("multi-agent-icl/outputs/summary.json"))

    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    if args.full_test:
        args.use_sampled_test = False

    sample_indices = icl.load_indices(args.sample_indices) if args.sample_indices.exists() else None
    test_indices = sample_indices if args.use_sampled_test and sample_indices is not None else None

    # # load train/test using the existing ICL loader
    # train_items = icl.load_split(
    #     split=args.train_split,
    #     dataset_name=args.dataset_name,
    #     dataset_config=args.dataset_config,
    #     local_path=args.train_file,
    #     indices=None,
    #     limit=None,
    # )
    # test_items = icl.load_split(
    #     split=args.test_split,
    #     dataset_name=args.dataset_name,
    #     dataset_config=args.dataset_config,
    #     local_path=args.test_file,
    #     indices=test_indices,
    #     limit=None,
    # )
    train_items = load_split_raw(
        split=args.train_split,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        local_path=args.train_file,
        indices=None,
        limit=None,
    )
    test_items = load_split_raw(
        split=args.test_split,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        local_path=args.test_file,
        indices=test_indices,
        limit=None,
    )

    if args.num_test_examples is not None:
        test_items = test_items[: args.num_test_examples]

    run_cfg = RunConfig(
        detector_model=args.detector_model,
        critic_model=args.critic_model,
        editor_model=args.editor_model,
        n_best=args.n_best,
        reasoning_effort=args.reasoning_effort,
    )
    
    icl_cfg = ICLConfig(
        mode=args.icl_mode,
        k_shot=args.k_shot,
        embedding_model=args.embedding_model,
        cache_dir=args.cache_dir,
        use_detector_icl=not args.disable_detector_icl,
        use_editor_icl=not args.disable_editor_icl,
        use_critic_icl=args.enable_critic_icl,
    )

    client = OpenAI()

    train_embeddings = None
    if icl_cfg.mode == "dynamic":
        train_embeddings = icl.embed_items(
            items=train_items,
            embedding_model=icl_cfg.embedding_model,
            cache_dir=icl_cfg.cache_dir,
            cache_prefix="train",
        )

    if args.out is None:
        ts = icl.latest_timestamp()
        args.out = Path(f"multi-agent-icl/outputs/medec_multi_agent_icl_{args.icl_mode}_{ts}.txt")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(test_items, start=1):
            try:
                line = run_one_with_icl(
                    client=client,
                    row=row,
                    run_cfg=run_cfg,
                    icl_cfg=icl_cfg,
                    train_items=train_items,
                    train_embeddings=train_embeddings,
                )
            except Exception as exc:
                print(f"[WARN] Failed on {row.get('text_id', 'UNKNOWN')}: {exc}")
                line = ma.to_submission_line(row.get("text_id", "UNKNOWN"), 0, -1, "NA")

            handle.write(line + "\n")
            print(f"[{idx}/{len(test_items)}] {line}")
            time.sleep(0.05)

    predictions = icl.parse_submission_file(args.out)
    metrics = icl.evaluate_predictions(predictions, test_items)

    summary = {
        "config": {
            "dataset_name": args.dataset_name,
            "dataset_config": args.dataset_config,
            "train_split": args.train_split,
            "test_split": args.test_split,
            "n_train": len(train_items),
            "n_test": len(test_items),
            "detector_model": args.detector_model,
            "critic_model": args.critic_model,
            "editor_model": args.editor_model,
            "n_best": args.n_best,
            "icl_mode": args.icl_mode,
            "k_shot": args.k_shot,
            "embedding_model": args.embedding_model,
            "use_detector_icl": icl_cfg.use_detector_icl,
            "use_editor_icl": icl_cfg.use_editor_icl,
            "use_critic_icl": icl_cfg.use_critic_icl,
        },
        "output_path": str(args.out),
        "metrics": metrics,
    }

    icl.write_json(args.summary_json, summary)

    print("\n[RESULT]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())