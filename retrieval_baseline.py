"""Reproducible, offline BM25 smoke tests for the Wikipedia passage dataset.

This lexical baseline validates corpus preparation and evidence retrieval. It does
not create dense embeddings, generate answers, or establish final QA performance.
Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parent
SCHEMA_VERSION = 1
METHOD = "bm25_lexical_smoke_test"
TOKENIZATION = "Unicode alphanumeric words, casefolded, fixed English stop words; no stemming"
STOP_WORDS = frozenset(
    "a an and are as at be been being but by can could did do does for from had has "
    "have how i if in into is it its may of on or should than that the their them "
    "there these they this those to was were what when where which who whom why "
    "will with would you your".split()
)


def tokenize(text: str) -> list[str]:
    """Preserve scientific terms and numbers without a downloaded tokenizer."""
    return [word for word in re.findall(r"[^\W_]+", text.casefold()) if word not in STOP_WORDS]


def corpus_digest(chunks: list[dict[str, Any]]) -> str:
    canonical = json.dumps(chunks, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    """Replace an artifact atomically so interrupted runs cannot leave partial JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = stream.name
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def validate_chunks(chunks: Any) -> list[dict[str, Any]]:
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("The corpus must contain at least one chunk.")
    seen: set[str] = set()
    for position, chunk in enumerate(chunks, 1):
        if not isinstance(chunk, dict):
            raise ValueError(f"Chunk {position} must be an object.")
        for key in ("chunk_id", "document_id", "text"):
            if not isinstance(chunk.get(key), str) or not chunk[key].strip():
                raise ValueError(f"Chunk {position} has an empty or invalid {key}.")
        if chunk["chunk_id"] in seen:
            raise ValueError(f"Duplicate chunk_id: {chunk['chunk_id']}")
        seen.add(chunk["chunk_id"])
    return chunks


@dataclass
class BM25Index:
    chunks: list[dict[str, Any]]
    k1: float = 1.5
    b: float = 0.75
    corpus_sha256: str = ""
    lengths: list[int] = field(init=False, repr=False)
    postings: dict[str, list[tuple[int, int]]] = field(init=False, repr=False)
    average_length: float = field(init=False)

    def __post_init__(self) -> None:
        validate_chunks(self.chunks)
        if not math.isfinite(self.k1) or self.k1 <= 0:
            raise ValueError("BM25 k1 must be finite and greater than zero.")
        if not math.isfinite(self.b) or not 0 <= self.b <= 1:
            raise ValueError("BM25 b must be between zero and one.")
        digest = corpus_digest(self.chunks)
        if self.corpus_sha256 and self.corpus_sha256 != digest:
            raise ValueError("Corpus digest mismatch: the saved chunk data changed or is corrupt.")
        self.corpus_sha256 = digest
        self.lengths = []
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, chunk in enumerate(self.chunks):
            counts = Counter(tokenize(chunk["text"]))
            self.lengths.append(sum(counts.values()))
            for term, count in counts.items():
                postings[term].append((index, count))
        self.postings = dict(postings)
        self.average_length = sum(self.lengths) / len(self.lengths)
        if self.average_length == 0:
            raise ValueError("The corpus contains no searchable lexical terms.")

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "method": METHOD,
            "status": "lexical_baseline_only_not_dense_RAG",
            "tokenization": TOKENIZATION,
            "stop_words": sorted(STOP_WORDS),
            "parameters": {"k1": self.k1, "b": self.b},
            "corpus_sha256": self.corpus_sha256,
            "chunk_count": len(self.chunks),
            "document_count": len({chunk["document_id"] for chunk in self.chunks}),
            "average_searchable_terms_per_chunk": self.average_length,
            "chunks": self.chunks,
        }


def build_index(chunks_path: Path | str, output_path: Path | str, k1: float = 1.5, b: float = 0.75) -> BM25Index:
    chunks = []
    with Path(chunks_path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    chunks.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid chunk JSON on line {line_number}: {exc.msg}") from exc
    index = BM25Index(chunks, k1=k1, b=b)
    write_json(Path(output_path), index.payload())
    return index


def load_index(path: Path | str) -> BM25Index:
    with Path(path).open(encoding="utf-8-sig") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported or missing BM25 index schema version.")
    if payload.get("method") != METHOD:
        raise ValueError("This file is not a BM25 lexical smoke-test index.")
    if payload.get("tokenization") != TOKENIZATION or payload.get("stop_words") != sorted(STOP_WORDS):
        raise ValueError("Tokenizer settings differ from the saved index. Rebuild the index.")
    if not isinstance(payload.get("corpus_sha256"), str) or len(payload["corpus_sha256"]) != 64:
        raise ValueError("Saved index has no valid corpus digest.")
    parameters = payload.get("parameters", {})
    index = BM25Index(
        payload.get("chunks"),
        k1=float(parameters.get("k1", 1.5)),
        b=float(parameters.get("b", 0.75)),
        corpus_sha256=payload["corpus_sha256"],
    )
    if payload.get("chunk_count") != len(index.chunks):
        raise ValueError("Saved chunk count differs from its chunk data.")
    return index


def search(index: BM25Index, question: str, top_k: int = 5) -> list[dict[str, Any]]:
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    if not isinstance(question, str):
        raise ValueError("The query must be text.")
    scores: dict[int, float] = defaultdict(float)
    count = len(index.chunks)
    for term in sorted(set(tokenize(question))):
        posting = index.postings.get(term, ())
        if not posting:
            continue
        document_frequency = len(posting)
        inverse_frequency = math.log(1 + (count - document_frequency + 0.5) / (document_frequency + 0.5))
        for chunk_number, frequency in posting:
            normalization = 1 - index.b + index.b * index.lengths[chunk_number] / index.average_length
            scores[chunk_number] += inverse_frequency * frequency * (index.k1 + 1) / (frequency + index.k1 * normalization)
    ranked = sorted(scores, key=lambda number: (-scores[number], index.chunks[number]["chunk_id"]))[:top_k]
    return [
        {
            "rank": rank,
            "score": round(scores[number], 8),
            **{key: index.chunks[number].get(key, "") for key in ("chunk_id", "document_id", "title", "url", "text")},
        }
        for rank, number in enumerate(ranked, 1)
    ]


def evaluate(index: BM25Index, questions_payload: dict[str, Any]) -> dict[str, Any]:
    questions = questions_payload.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("Evaluation requires a nonempty questions list.")
    chunk_documents = {chunk["chunk_id"]: chunk["document_id"] for chunk in index.chunks}
    seen: set[str] = set()
    results = []
    for question in questions:
        question_id = question.get("id")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError("Each evaluation question needs a nonempty id.")
        if question_id in seen:
            raise ValueError(f"Duplicate question id: {question_id}")
        seen.add(question_id)
        if not isinstance(question.get("question"), str) or not question["question"].strip():
            raise ValueError(f"Question {question_id} has no query text.")
        if not isinstance(question.get("answerable"), bool):
            raise ValueError(f"Question {question_id} must explicitly declare answerable true or false.")
        gold = question.get("gold_chunk_ids") or []
        if not isinstance(gold, list) or any(not isinstance(value, str) for value in gold):
            raise ValueError(f"Question {question_id} has invalid gold_chunk_ids.")
        unknown = set(gold) - chunk_documents.keys()
        if unknown:
            raise ValueError(f"Question {question_id} references missing chunks: {sorted(unknown)}")
        if question.get("document_id") and any(chunk_documents[value] != question["document_id"] for value in gold):
            raise ValueError(f"Question {question_id} has evidence from another document.")
        if not question["answerable"] and gold:
            raise ValueError(f"Unanswerable question {question_id} must not have gold evidence.")
        eligible = question["answerable"] and question.get("evidence_status") == "mapped" and bool(gold)
        matches = search(index, question["question"], 5)
        record = {
            "id": question_id,
            "split": question.get("split", "unspecified"),
            "answerable": question["answerable"],
            "review_status": question.get("review_status", "unspecified"),
            "included_in_evidence_hit_denominator": eligible,
            "gold_chunk_ids": gold,
            "top_results": [{key: match[key] for key in ("rank", "score", "chunk_id", "document_id")} for match in matches],
        }
        if eligible:
            record["evidence_hit"] = {str(k): bool(set(gold) & {match["chunk_id"] for match in matches[:k]}) for k in (1, 3, 5)}
        elif question["answerable"]:
            record["exclusion_reason"] = "answerable_question_has_no_complete_mapped_evidence"
        else:
            record["exclusion_reason"] = "corpus_unanswerable_question_no_evidence_hit_metric"
        results.append(record)

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        included = [item for item in items if item["included_in_evidence_hit_denominator"]]
        summary: dict[str, Any] = {"question_count": len(items), "mapped_answerable_denominator": len(included)}
        for k in (1, 3, 5):
            hits = sum(item["evidence_hit"][str(k)] for item in included)
            summary[f"evidence_hit_at_{k}"] = {"hits": hits, "denominator": len(included), "rate": round(hits / len(included), 6) if included else None}
        return summary

    return {
        "status": "draft_retrieval_smoke_test_not_final_benchmark",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": METHOD,
        "corpus_sha256": index.corpus_sha256,
        "question_set_status": questions_payload.get("status", "unspecified"),
        "limitations": [
            "Lexical retrieval only. Dense embeddings and answer generation have not been evaluated.",
            "Draft questions and evidence require human review before final test freezing.",
            "Evidence hit measures retrieval of an annotated complete supporting chunk, not answer correctness.",
            "Scores for unanswerable questions do not establish abstention quality or accuracy.",
        ],
        "overall": summarize(results),
        "by_split": {split: summarize([item for item in results if item["split"] == split]) for split in sorted({item["split"] for item in results})},
        "unanswerable_retrieval_scores": [
            {"id": item["id"], "split": item["split"], "top_score": item["top_results"][0]["score"] if item["top_results"] else None, "retrieved_chunks": len(item["top_results"])}
            for item in results if not item["answerable"]
        ],
        "questions": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="Create a portable lexical index from processed chunks.")
    build.add_argument("--chunks", type=Path, default=ROOT / "data/processed/chunks.jsonl")
    build.add_argument("--output", type=Path, default=ROOT / "data/index/bm25.json")
    build.add_argument("--k1", type=float, default=1.5)
    build.add_argument("--b", type=float, default=0.75)
    query = subparsers.add_parser("query", help="Find passages without network access or model downloads.")
    query.add_argument("--index", type=Path, default=ROOT / "data/index/bm25.json")
    query.add_argument("--question", required=True)
    query.add_argument("--top-k", type=int, default=5)
    query.add_argument("--output", type=Path)
    evaluation = subparsers.add_parser("evaluate", help="Check annotated evidence retrieval on draft questions.")
    evaluation.add_argument("--index", type=Path, default=ROOT / "data/index/bm25.json")
    evaluation.add_argument("--questions", type=Path, default=ROOT / "evaluation/questions_mapped.json")
    evaluation.add_argument("--output", type=Path, default=ROOT / "reports/retrieval_smoke_test.json")
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            index = build_index(args.chunks, args.output, args.k1, args.b)
            print(json.dumps({"method": METHOD, "chunks": len(index.chunks), "documents": len({item["document_id"] for item in index.chunks}), "corpus_sha256": index.corpus_sha256, "output": str(args.output)}, indent=2))
        elif args.command == "query":
            index = load_index(args.index)
            output = {"method": METHOD, "question": args.question, "results": search(index, args.question, args.top_k)}
            if args.output:
                write_json(args.output, output)
            print(json.dumps(output, ensure_ascii=False, indent=2))
        else:
            index = load_index(args.index)
            with args.questions.open(encoding="utf-8-sig") as stream:
                report = evaluate(index, json.load(stream))
            write_json(args.output, report)
            print(json.dumps({"status": report["status"], "overall": report["overall"], "by_split": report["by_split"], "output": str(args.output)}, indent=2))
    except (ValueError, OSError, TypeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
