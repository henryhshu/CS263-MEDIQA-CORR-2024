#!/usr/bin/env python3
"""Run baseline, fixed-shot, and retrieval-based ICL experiments for MEDEC."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request


DEFAULT_DATASET = "mkieffer/MEDEC"
DEFAULT_DATASET_CONFIG = "default"
DEFAULT_MODEL = "gpt-4.1"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_MODES = ("baseline", "fixed", "dynamic")
DEFAULT_SAMPLE_INDICES = Path("baseline-experiment/sampled_test_indices.json")
DEFAULT_BASELINE_OUTPUT = Path("baseline-experiment/outputs/medec-ms_gpt4_1_results_20260205_180349.txt")
SYSTEM_PROMPT = """
The following is a medical narrative about a patient. You are a skilled medical doctor reviewing the clinical text. The text is either correct or contains one error.
The text has a sentence per line. Each line starts with the sentence ID, followed by a pipe character then the sentence to check. Check every sentence of the text.
If the text is correct return the following output: CORRECT. If the text has a medical error, return the sentence id of the sentence containing the error,
followed by a space, and a corrected version of the sentence.
""".strip()


def eprint(*parts: Any) -> None:
    print(*parts, file=sys.stderr)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def maybe_int(value: Any, default: int = -1) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    text = normalize_whitespace(str(value))
    if not text:
        return default
    try:
        return int(float(text))
    except ValueError:
        return default


def first_present(mapping: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def normalize_corrected_sentence(value: Any) -> str:
    if value is None:
        return "NA"
    text = normalize_whitespace(str(value))
    if not text or text.upper() == "NAN":
        return "NA"
    if text == "NA.":
        return "NA"
    return text


def canonicalize_item(item: dict[str, Any], fallback_idx: int | None = None) -> dict[str, Any]:
    text_id = first_present(item, ["text_id", "Text ID", "id", "uid", "doc_id"], None)
    if text_id is None:
        if fallback_idx is None:
            raise ValueError("Item is missing a usable text identifier.")
        text_id = f"idx-{fallback_idx}"

    sentences = first_present(item, ["sentences", "Sentences", "text", "Text"], "")
    error_flag = maybe_int(first_present(item, ["error_flag", "Error Flag"], 0), 0)
    error_sentence_id = maybe_int(first_present(item, ["error_sentence_id", "Error Sentence ID"], -1), -1)
    corrected_sentence = normalize_corrected_sentence(
        first_present(item, ["corrected_sentence", "Corrected Sentence"], "NA")
    )

    return {
        "text_id": str(text_id),
        "sentences": str(sentences),
        "error_flag": error_flag,
        "error_sentence_id": error_sentence_id,
        "corrected_sentence": corrected_sentence,
    }


def load_indices(path: Path) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "indices" in payload:
        payload = payload["indices"]
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list or an object with 'indices' in {path}")
    return [int(x) for x in payload]


def load_local_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if "data" in payload and isinstance(payload["data"], list):
            payload = payload["data"]
        else:
            payload = [payload]
    if not isinstance(payload, list):
        raise ValueError(f"Unsupported JSON structure in {path}")
    return [canonicalize_item(item, idx) for idx, item in enumerate(payload)]


def load_local_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            items.append(canonicalize_item(json.loads(line), idx))
    return items


def load_local_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [canonicalize_item(row, idx) for idx, row in enumerate(reader)]


def load_local_dataset(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return load_local_json(path)
    if suffix == ".jsonl":
        return load_local_jsonl(path)
    if suffix == ".csv":
        return load_local_csv(path)
    raise ValueError(f"Unsupported dataset file type: {path}")


def http_json(
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    payload: dict[str, Any] | None = None,
    timeout: int = 120,
    max_retries: int = 4,
) -> dict[str, Any]:
    body = None
    request_headers = dict(headers or {})
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    for attempt in range(max_retries + 1):
        request = urllib_request.Request(url=url, data=body, method=method.upper(), headers=request_headers)
        try:
            with urllib_request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib_error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            if exc.code in {429, 500, 502, 503, 504} and attempt < max_retries:
                sleep_seconds = min(2 ** attempt, 20)
                eprint(f"Retrying {url} after HTTP {exc.code} in {sleep_seconds}s")
                time.sleep(sleep_seconds)
                continue
            raise RuntimeError(f"HTTP {exc.code} for {url}: {error_body}") from exc
        except urllib_error.URLError as exc:
            if attempt < max_retries:
                sleep_seconds = min(2 ** attempt, 20)
                eprint(f"Retrying {url} after network error in {sleep_seconds}s: {exc}")
                time.sleep(sleep_seconds)
                continue
            raise RuntimeError(f"Request failed for {url}: {exc}") from exc
    raise RuntimeError(f"Request failed for {url}: retries exhausted")


def resolve_dataset_config(dataset_name: str, preferred_config: str | None) -> str:
    if preferred_config:
        return preferred_config
    url = f"https://datasets-server.huggingface.co/splits?dataset={urllib_parse.quote(dataset_name)}"
    payload = http_json("GET", url)
    splits = payload.get("splits", [])
    if not splits:
        raise RuntimeError(f"Could not resolve config for dataset {dataset_name}")
    return str(splits[0]["config"])


def load_split_via_hf_rows(
    dataset_name: str,
    dataset_config: str,
    split: str,
    page_size: int = 100,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    offset = 0
    while True:
        params = {
            "dataset": dataset_name,
            "config": dataset_config,
            "split": split,
            "offset": str(offset),
            "length": str(page_size),
        }
        url = "https://datasets-server.huggingface.co/rows?" + urllib_parse.urlencode(params)
        payload = http_json("GET", url)
        rows = payload.get("rows", [])
        if not rows:
            break
        for row in rows:
            items.append(canonicalize_item(row["row"], len(items)))
        if len(rows) < page_size:
            break
        offset += len(rows)
    return items


def fetch_hf_row_range(
    dataset_name: str,
    dataset_config: str,
    split: str,
    offset: int,
    length: int,
) -> list[dict[str, Any]]:
    params = {
        "dataset": dataset_name,
        "config": dataset_config,
        "split": split,
        "offset": str(offset),
        "length": str(length),
    }
    url = "https://datasets-server.huggingface.co/rows?" + urllib_parse.urlencode(params)
    payload = http_json("GET", url)
    rows = payload.get("rows", [])
    return [canonicalize_item(row["row"], row.get("row_idx")) for row in rows]


def contiguous_ranges(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    sorted_indices = sorted(set(indices))
    ranges: list[tuple[int, int]] = []
    start = sorted_indices[0]
    end = start
    for index in sorted_indices[1:]:
        if index == end + 1:
            end = index
            continue
        ranges.append((start, end - start + 1))
        start = end = index
    ranges.append((start, end - start + 1))
    return ranges


def load_split_via_hf_rows_targeted(
    dataset_name: str,
    dataset_config: str,
    split: str,
    indices: list[int] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if indices is not None:
        items: list[dict[str, Any]] = []
        for offset, length in contiguous_ranges(indices):
            items.extend(fetch_hf_row_range(dataset_name, dataset_config, split, offset, length))
            time.sleep(0.05)
        item_by_index = {index: item for index, item in zip(sorted(set(indices)), items)}
        return [item_by_index[index] for index in indices]

    if limit is not None:
        return fetch_hf_row_range(dataset_name, dataset_config, split, 0, limit)

    return load_split_via_hf_rows(dataset_name, dataset_config, split)


def load_hf_dataset_split(
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    indices: list[int] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore

        if dataset_config:
            dataset = load_dataset(dataset_name, dataset_config, split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)
        if indices is not None:
            dataset = dataset.select(indices)
        elif limit is not None:
            dataset = dataset.select(range(min(limit, len(dataset))))
        return [canonicalize_item(item, idx) for idx, item in enumerate(dataset)]
    except Exception as exc:
        eprint(f"Falling back to the Hugging Face dataset server for split '{split}': {exc}")
        resolved_config = resolve_dataset_config(dataset_name, dataset_config or DEFAULT_DATASET_CONFIG)
        return load_split_via_hf_rows_targeted(
            dataset_name=dataset_name,
            dataset_config=resolved_config,
            split=split,
            indices=indices,
            limit=limit,
        )


def load_split(
    split: str,
    dataset_name: str,
    dataset_config: str | None,
    local_path: Path | None,
    indices: list[int] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if local_path is not None:
        items = load_local_dataset(local_path)
        if indices is not None:
            items = select_indices(items, indices)
        if limit is not None:
            items = items[:limit]
        return items
    return load_hf_dataset_split(dataset_name, dataset_config, split, indices=indices, limit=limit)


def select_indices(items: list[dict[str, Any]], indices: list[int]) -> list[dict[str, Any]]:
    return [items[idx] for idx in indices]


def strip_sentence_ids(sentences: str) -> str:
    lines = []
    for raw_line in str(sentences).splitlines():
        lines.append(re.sub(r"^\s*\d+\s*\|\s*", "", raw_line).strip())
    return "\n".join(line for line in lines if line)


def retrieval_text(item: dict[str, Any]) -> str:
    return strip_sentence_ids(item["sentences"])


def gold_response(item: dict[str, Any]) -> str:
    if item["error_flag"] == 0:
        return "CORRECT."
    return f'{item["error_sentence_id"]} {item["corrected_sentence"]}'


def build_messages(query_item: dict[str, Any], examples: list[dict[str, Any]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example in examples:
        messages.append({"role": "user", "content": example["sentences"]})
        messages.append({"role": "assistant", "content": gold_response(example)})
    messages.append({"role": "user", "content": query_item["sentences"]})
    return messages


def escape_for_double_quotes(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def parse_model_output(output_text: str) -> tuple[int, int, str]:
    text = normalize_whitespace(output_text)
    if not text:
        return 0, -1, "NA"

    if text.upper().rstrip(".") == "CORRECT":
        return 0, -1, "NA"

    parts = text.split(None, 1)
    if not parts:
        return 0, -1, "NA"

    try:
        sentence_id = int(parts[0])
    except ValueError:
        return 0, -1, "NA"

    corrected = normalize_whitespace(parts[1]) if len(parts) > 1 else "NA"
    return 1, sentence_id, corrected or "NA"


def submission_line(text_id: str, error_flag: int, sentence_id: int, corrected_sentence: str) -> str:
    if error_flag == 0:
        return f"{text_id} 0 -1 NA"
    escaped = escape_for_double_quotes(corrected_sentence)
    return f'{text_id} 1 {sentence_id} "{escaped}"'


def parse_submission_file(path: Path) -> dict[str, dict[str, Any]]:
    pattern = re.compile(r"^([A-Za-z0-9\-_]+)\s+([01])\s+(-?\d+)\s+(.+)$")
    record_start_pattern = re.compile(r"^[A-Za-z0-9\-_]+\s+[01]\s+-?\d+\s+")
    predictions: dict[str, dict[str, Any]] = {}
    records: list[str] = []
    current_record = ""

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            if record_start_pattern.match(line):
                if current_record:
                    records.append(current_record)
                current_record = line
                continue

            if not current_record:
                raise ValueError(f"Invalid submission line {line_number} in {path}: {line}")

            # Some notebook-generated outputs wrap quoted corrections across lines.
            current_record += " " + line

    if current_record:
        records.append(current_record)

    for record_number, line in enumerate(records, start=1):
        match = pattern.match(line)
        if not match:
            raise ValueError(f"Invalid submission record {record_number} in {path}: {line}")
        text_id, flag_text, sid_text, correction_text = match.groups()
        correction_text = correction_text.strip()
        while correction_text.startswith('"') and len(correction_text) > 1:
            correction_text = correction_text[1:]
        while correction_text.endswith('"') and len(correction_text) > 1:
            correction_text = correction_text[:-1]
        error_flag = int(flag_text)
        predictions[text_id] = {
            "text_id": text_id,
            "error_flag": error_flag,
            "error_sentence_id": int(sid_text),
            "corrected_sentence": "NA" if error_flag == 0 else correction_text,
        }
    return predictions


def tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def rouge1_f1(reference: str, prediction: str) -> float:
    ref_tokens = tokens(reference)
    pred_tokens = tokens(prediction)
    if not ref_tokens or not pred_tokens:
        return 0.0
    ref_counts = Counter(ref_tokens)
    pred_counts = Counter(pred_tokens)
    overlap = sum(min(ref_counts[token], pred_counts[token]) for token in ref_counts)
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current.append(prev[j - 1] + 1)
            else:
                current.append(max(prev[j], current[-1]))
        prev = current
    return prev[-1]


def rouge_l_f1(reference: str, prediction: str) -> float:
    ref_tokens = tokens(reference)
    pred_tokens = tokens(prediction)
    if not ref_tokens or not pred_tokens:
        return 0.0
    lcs = lcs_length(ref_tokens, pred_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_predictions(
    predictions: dict[str, dict[str, Any]],
    references: list[dict[str, Any]],
) -> dict[str, float]:
    total = len(references)
    flag_matches = 0
    sentence_matches = 0
    exact_matches = 0
    rouge1_subset: list[float] = []
    rouge_l_subset: list[float] = []
    rouge1_composite = 0.0
    rouge_l_composite = 0.0

    for reference in references:
        predicted = predictions.get(reference["text_id"], None)
        predicted_flag = predicted["error_flag"] if predicted else 0
        predicted_sid = predicted["error_sentence_id"] if predicted else -1
        predicted_correction = normalize_corrected_sentence(
            predicted["corrected_sentence"] if predicted else "NA"
        )
        reference_correction = normalize_corrected_sentence(reference["corrected_sentence"])

        if predicted_flag == reference["error_flag"]:
            flag_matches += 1
        if predicted_sid == reference["error_sentence_id"]:
            sentence_matches += 1
        if predicted_correction == reference_correction:
            exact_matches += 1

        if reference_correction == "NA" and predicted_correction == "NA":
            rouge1_composite += 1.0
            rouge_l_composite += 1.0
            continue

        if reference_correction == "NA" or predicted_correction == "NA":
            continue

        rouge1 = rouge1_f1(reference_correction, predicted_correction)
        rouge_l = rouge_l_f1(reference_correction, predicted_correction)
        rouge1_subset.append(rouge1)
        rouge_l_subset.append(rouge_l)
        rouge1_composite += rouge1
        rouge_l_composite += rouge_l

    subset_count = len(rouge1_subset)
    return {
        "n_texts": float(total),
        "error_flag_accuracy": flag_matches / total if total else 0.0,
        "error_sentence_accuracy": sentence_matches / total if total else 0.0,
        "correction_exact_match_all": exact_matches / total if total else 0.0,
        "rouge1_f1_error_subset": sum(rouge1_subset) / subset_count if subset_count else 0.0,
        "rougeL_f1_error_subset": sum(rouge_l_subset) / subset_count if subset_count else 0.0,
        "rouge1_f1_composite": rouge1_composite / total if total else 0.0,
        "rougeL_f1_composite": rouge_l_composite / total if total else 0.0,
    }


def stable_hash(payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def read_json_if_exists(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def openai_base_url() -> str:
    return os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")


def openai_headers() -> dict[str, str]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return {
        "Authorization": f"Bearer {api_key}",
    }


def extract_response_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str) and payload["output_text"].strip():
        return payload["output_text"]

    chunks: list[str] = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                text = content.get("text") or content.get("value") or ""
                if text:
                    chunks.append(text)
    return "\n".join(chunks).strip()


def call_openai_response(
    model: str,
    messages: list[dict[str, str]],
    reasoning_effort: str | None = None,
    temperature: float | None = None,
) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "input": messages,
    }
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}
    if temperature is not None:
        payload["temperature"] = temperature
    response = http_json(
        "POST",
        f"{openai_base_url()}/responses",
        headers=openai_headers(),
        payload=payload,
        timeout=300,
    )
    text = extract_response_text(response)
    if not text:
        raise RuntimeError(f"OpenAI response did not contain text: {json.dumps(response)[:500]}")
    return text


def call_openai_embedding(model: str, text: str) -> list[float]:
    return call_openai_embeddings(model, [text])[0]


def call_openai_embeddings(model: str, texts: list[str]) -> list[list[float]]:
    payload = {
        "model": model,
        "input": texts,
    }
    response = http_json(
        "POST",
        f"{openai_base_url()}/embeddings",
        headers=openai_headers(),
        payload=payload,
        timeout=300,
    )
    data = response.get("data", [])
    if not data or len(data) != len(texts):
        raise RuntimeError(f"Embedding response was empty: {json.dumps(response)[:500]}")
    embeddings: list[list[float]] = []
    for row in data:
        embedding = row.get("embedding")
        if not isinstance(embedding, list):
            raise RuntimeError(f"Embedding response was malformed: {json.dumps(response)[:500]}")
        embeddings.append([float(value) for value in embedding])
    return embeddings


def cached_embedding(cache_dir: Path, namespace: str, model: str, cache_id: str, text: str) -> list[float]:
    cache_path = cache_dir / f"{namespace}_{model}_{cache_id}.json"
    cached = read_json_if_exists(cache_path)
    if isinstance(cached, list):
        return [float(value) for value in cached]
    embedding = call_openai_embedding(model, text)
    write_json(cache_path, embedding)
    return embedding


def cosine_similarity(a: list[float], b: list[float]) -> float:
    numerator = sum(x * y for x, y in zip(a, b))
    a_norm = math.sqrt(sum(x * x for x in a))
    b_norm = math.sqrt(sum(y * y for y in b))
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return numerator / (a_norm * b_norm)


def cache_key_for_examples(items: list[dict[str, Any]]) -> str:
    return stable_hash([item["text_id"] for item in items])


def embed_items(
    items: list[dict[str, Any]],
    embedding_model: str,
    cache_dir: Path,
    cache_prefix: str,
    batch_size: int = 64,
) -> dict[str, list[float]]:
    cache_path = cache_dir / f"{cache_prefix}_{embedding_model}_{cache_key_for_examples(items)}.json"
    cached = read_json_if_exists(cache_path)
    if isinstance(cached, dict):
        return {str(key): [float(v) for v in value] for key, value in cached.items()}

    embeddings: dict[str, list[float]] = {}
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        batch_embeddings = call_openai_embeddings(
            embedding_model,
            [retrieval_text(item) for item in batch],
        )
        for item, embedding in zip(batch, batch_embeddings):
            embeddings[item["text_id"]] = embedding
        eprint(f"Embedded {min(start + len(batch), len(items))}/{len(items)} items for {cache_prefix}")
        time.sleep(0.1)

    write_json(cache_path, embeddings)
    return embeddings


def nearest_examples(
    train_items: list[dict[str, Any]],
    train_embeddings: dict[str, list[float]],
    query_embedding: list[float],
    k: int,
) -> list[dict[str, Any]]:
    scored: list[tuple[float, dict[str, Any]]] = []
    for item in train_items:
        train_embedding = train_embeddings[item["text_id"]]
        scored.append((cosine_similarity(query_embedding, train_embedding), item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored[:k]]


def latest_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_mode(
    mode: str,
    test_items: list[dict[str, Any]],
    train_items: list[dict[str, Any]],
    output_dir: Path,
    model: str,
    k_shot: int,
    embedding_model: str,
    cache_dir: Path,
    reasoning_effort: str | None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"medec_{mode}_{model.replace('.', '_')}_{latest_timestamp()}.txt"

    fixed_examples = train_items[:k_shot] if mode == "fixed" else []
    train_embeddings: dict[str, list[float]] = {}
    if mode == "dynamic":
        train_embeddings = embed_items(train_items, embedding_model, cache_dir, "train")

    with out_path.open("w", encoding="utf-8") as handle:
        for idx, item in enumerate(test_items, start=1):
            examples: list[dict[str, Any]]
            if mode == "baseline":
                examples = []
            elif mode == "fixed":
                examples = fixed_examples
            elif mode == "dynamic":
                query_embedding = cached_embedding(
                    cache_dir=cache_dir,
                    namespace="query",
                    model=embedding_model,
                    cache_id=item["text_id"],
                    text=retrieval_text(item),
                )
                examples = nearest_examples(train_items, train_embeddings, query_embedding, k_shot)
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            messages = build_messages(item, examples)
            output_text = call_openai_response(
                model=model,
                messages=messages,
                reasoning_effort=reasoning_effort,
            )
            error_flag, sentence_id, corrected_sentence = parse_model_output(output_text)
            line = submission_line(item["text_id"], error_flag, sentence_id, corrected_sentence)
            handle.write(line + "\n")

            eprint(f"[{mode}] {idx}/{len(test_items)} {line}")
            time.sleep(0.05)

    return out_path


def format_metrics(metrics: dict[str, float]) -> str:
    ordered_keys = [
        "error_flag_accuracy",
        "error_sentence_accuracy",
        "correction_exact_match_all",
        "rouge1_f1_error_subset",
        "rougeL_f1_error_subset",
        "rouge1_f1_composite",
        "rougeL_f1_composite",
    ]
    return ", ".join(f"{key}={metrics[key]:.4f}" for key in ordered_keys)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run baseline, fixed-shot, and retrieval-based ICL experiments for MEDEC.",
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--test-file", type=Path)
    parser.add_argument("--sample-indices", type=Path, default=DEFAULT_SAMPLE_INDICES)
    parser.add_argument("--use-sampled-test", dest="use_sampled_test", action="store_true")
    parser.add_argument("--no-sampled-test", dest="use_sampled_test", action="store_false")
    parser.add_argument("--full-test", action="store_true")
    parser.add_argument("--num-test-examples", type=int)
    parser.add_argument("--modes", nargs="+", choices=list(DEFAULT_MODES), default=list(DEFAULT_MODES))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", choices=["minimal", "low", "medium", "high"])
    parser.add_argument("--k-shot", type=int, default=3)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--output-dir", type=Path, default=Path("in-context-learning/outputs"))
    parser.add_argument("--cache-dir", type=Path, default=Path("in-context-learning/cache"))
    parser.add_argument("--summary-json", type=Path, default=Path("in-context-learning/outputs/summary.json"))
    parser.add_argument("--existing-baseline-output", type=Path, default=DEFAULT_BASELINE_OUTPUT)
    parser.add_argument("--skip-existing-baseline", action="store_true")
    parser.set_defaults(use_sampled_test=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    if args.full_test:
        args.use_sampled_test = False

    sample_indices = load_indices(args.sample_indices) if args.sample_indices.exists() else None
    train_limit = None
    if "dynamic" not in args.modes:
        train_limit = args.k_shot

    test_indices = None
    if args.use_sampled_test and sample_indices is not None:
        test_indices = sample_indices

    train_items = load_split(
        args.train_split,
        args.dataset_name,
        args.dataset_config,
        args.train_file,
        limit=train_limit,
    )
    test_items = load_split(
        args.test_split,
        args.dataset_name,
        args.dataset_config,
        args.test_file,
        indices=test_indices,
    )

    if args.num_test_examples is not None:
        test_items = test_items[: args.num_test_examples]

    run_results: dict[str, dict[str, Any]] = {
        "config": {
            "dataset_name": args.dataset_name,
            "dataset_config": args.dataset_config,
            "train_split": args.train_split,
            "test_split": args.test_split,
            "model": args.model,
            "embedding_model": args.embedding_model,
            "k_shot": args.k_shot,
            "n_train": len(train_items),
            "n_test": len(test_items),
        },
        "experiments": {},
    }

    if not args.skip_existing_baseline and args.existing_baseline_output and args.existing_baseline_output.exists():
        baseline_predictions = parse_submission_file(args.existing_baseline_output)
        baseline_metrics = evaluate_predictions(baseline_predictions, test_items)
        run_results["experiments"]["baseline_existing"] = {
            "path": str(args.existing_baseline_output),
            "metrics": baseline_metrics,
        }
        eprint(f"[baseline_existing] {format_metrics(baseline_metrics)}")

    for mode in args.modes:
        output_path = run_mode(
            mode=mode,
            test_items=test_items,
            train_items=train_items,
            output_dir=args.output_dir,
            model=args.model,
            k_shot=args.k_shot,
            embedding_model=args.embedding_model,
            cache_dir=args.cache_dir,
            reasoning_effort=args.reasoning_effort,
        )
        predictions = parse_submission_file(output_path)
        metrics = evaluate_predictions(predictions, test_items)
        run_results["experiments"][mode] = {
            "path": str(output_path),
            "metrics": metrics,
        }
        eprint(f"[{mode}] {format_metrics(metrics)}")

    write_json(args.summary_json, run_results)
    print(json.dumps(run_results, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
