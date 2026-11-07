"""Reproducible, network-free preprocessing of the local Wikipedia pilot.

Only Python's standard library is required. Evaluation labels never enter chunks.
Run ``python dataset_pipeline.py --help`` for commands.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics
import sys
import unicodedata
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parent
BUNDLED_SOURCE = ROOT / "data/raw/wikipedia_source.jsonl"
DEFAULT_SOURCE = BUNDLED_SOURCE if BUNDLED_SOURCE.exists() else ROOT.parent / "dl_project/datasets/wikipedia/wikipedia_slice_200.jsonl"
VERSION = "1.0"
WORD = re.compile(r"\S+")


@dataclass(frozen=True)
class Settings:
    min_words: int = 80
    chunk_words: int = 200
    overlap_words: int = 40
    max_articles: int = 500
    exclude_disambiguation: bool = True
    strip_end_matter: bool = True

    def validate(self):
        if self.min_words < 1 or self.max_articles < 1:
            raise ValueError("min_words and max_articles must be positive")
        if self.chunk_words < 2 or not 0 <= self.overlap_words < self.chunk_words:
            raise ValueError("Require chunk_words >= 2 and 0 <= overlap_words < chunk_words")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON at {path.name}:{line_no}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path.name}:{line_no}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No records in {path}")
    return rows


def normalize_text(text: str) -> str:
    """Conservative normalization. Preserve wording, case, symbols and paragraphs."""
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    text = "\n".join(re.sub(r"[^\S\n]+", " ", line).strip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def summary(numbers):
    if not numbers:
        return {"min": 0, "median": 0, "mean": 0, "max": 0, "total": 0}
    return {"min": min(numbers), "median": statistics.median(numbers),
            "mean": round(statistics.mean(numbers), 2), "max": max(numbers), "total": sum(numbers)}


def remove_end_matter(text: str, minimum_body_words: int = 80):
    """Trim only a standalone conventional end heading after a substantial body.

    The original snapshot remains intact. A heading near the beginning can be a
    contents entry, so ignore it instead of deleting the article's body.
    """
    pattern = re.compile(r"(?m)^(References|External links|Further reading|Bibliography) *$")
    for match in pattern.finditer(text):
        prefix = text[:match.start()].rstrip()
        if len(WORD.findall(prefix)) >= minimum_body_words:
            return prefix, match.group(1), len(WORD.findall(text[match.start():]))
    return text, None, 0


def prepare_documents(rows: list[dict], settings: Settings):
    settings.validate()
    retained, rejected = [], []
    seen_ids, seen_text = {}, set()
    quality = Counter()
    for index, row in enumerate(rows, 1):
        missing = [key for key in ("id", "url", "title", "text") if key not in row]
        if missing:
            raise ValueError(f"Record {index} missing required fields: {missing}")
        if not isinstance(row["id"], (str, int)) or isinstance(row["id"], bool):
            raise ValueError(f"Record {index} needs a string/integer id")
        if any(not isinstance(row[key], str) for key in ("url", "title", "text")):
            raise ValueError(f"Record {index} title, url and text must be strings")
        original_id = str(row["id"]).strip()
        title, text, url = normalize_text(row["title"]), normalize_text(row["text"]), row["url"].strip()
        normalized_text = text
        end_heading, removed_words = None, 0
        if settings.strip_end_matter:
            text, end_heading, removed_words = remove_end_matter(text, settings.min_words)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        words = len(WORD.findall(text))
        reasons = []
        if not original_id:
            reasons.append("empty_id")
        if not title:
            reasons.append("empty_title")
        parsed = urlsplit(url)
        if parsed.scheme not in ("https", "http") or parsed.hostname != "en.wikipedia.org":
            reasons.append("invalid_english_wikipedia_url")
        if words < settings.min_words:
            reasons.append("too_short")
        if settings.exclude_disambiguation and title.casefold().endswith("(disambiguation)"):
            reasons.append("disambiguation_title")
        if original_id in seen_ids:
            if seen_ids[original_id] != digest:
                raise ValueError(f"Conflicting text for duplicate article id {original_id}")
            reasons.append("duplicate_id")
        if digest in seen_text:
            reasons.append("duplicate_normalized_text")
        seen_ids[original_id] = digest
        seen_text.add(digest)
        if not reasons and len(retained) >= settings.max_articles:
            reasons.append("pilot_limit")
        quality["normalization_changed"] += normalized_text != row["text"]
        quality["end_matter_trimmed_articles"] += end_heading is not None
        quality["end_matter_removed_words"] += removed_words
        quality["replacement_character_articles"] += "\ufffd" in text
        quality["raw_html_marker_articles"] += bool(re.search(r"</?(?:div|p|table|script)\b", text, re.I))
        quality["list_title_articles"] += title.startswith("List of ")
        if reasons:
            rejected.append({"original_id": original_id, "title": title, "word_count": words, "reasons": reasons})
            continue
        retained.append({"document_id": f"wiki:{original_id}", "original_id": original_id,
                         "title": title, "url": url, "text": text, "word_count": words,
                         "text_sha256": digest, "source_row": index,
                         "end_matter_heading_removed": end_heading,
                         "end_matter_words_removed": removed_words})
    if not retained:
        raise ValueError("All source articles were rejected; review filters")
    return retained, rejected, dict(quality)


def make_chunks(documents: list[dict], settings: Settings) -> list[dict]:
    settings.validate()
    chunks = []
    for document in documents:
        spans = list(WORD.finditer(document["text"]))
        position, number = 0, 0
        while position < len(spans):
            end_word = min(position + settings.chunk_words, len(spans))
            start_char, end_char = spans[position].start(), spans[end_word - 1].end()
            text = document["text"][start_char:end_char]
            identity = f"{document['text_sha256']}:{settings.chunk_words}:{settings.overlap_words}:{start_char}:{end_char}"
            suffix = hashlib.sha256(identity.encode()).hexdigest()[:12]
            chunks.append({"chunk_id": f"{document['document_id']}:c{number:04d}:{suffix}",
                           "document_id": document["document_id"], "title": document["title"],
                           "url": document["url"], "text": text,
                           "word_count": end_word - position, "char_start": start_char,
                           "char_end": end_char, "word_start": position, "word_end": end_word,
                           "section_index": None, "chunk_index": number})
            if end_word == len(spans):
                break
            position = end_word - settings.overlap_words
            number += 1
    return chunks


def map_questions(payload: dict, documents: list[dict], chunks: list[dict]):
    result = dict(payload)
    result["questions"] = []
    result["status"] = "draft_requires_human_review"
    docs = {d["original_id"]: d for d in documents}
    by_doc = {}
    for chunk in chunks:
        by_doc.setdefault(chunk["document_id"], []).append(chunk)
    seen, split_docs = set(), {"development": set(), "test": set()}
    for original in payload["questions"]:
        q = dict(original)
        if q["id"] in seen:
            raise ValueError(f"Duplicate question id: {q['id']}")
        seen.add(q["id"])
        if q.get("split") not in split_docs or not isinstance(q.get("answerable"), bool):
            raise ValueError(f"Invalid question split/answerable: {q['id']}")
        q["gold_chunk_ids"] = []
        q["document_id"] = None
        q["evidence_status"] = "not_applicable"
        if q["answerable"]:
            if not q.get("reference_answers") or not q.get("evidence_quote"):
                raise ValueError(f"Missing answer/evidence for {q['id']}")
            doc = docs.get(str(q.get("source_document_id")))
            q["evidence_status"] = "source_not_retained"
            if doc:
                q["document_id"] = doc["document_id"]
                split_docs[q["split"]].add(doc["document_id"])
                quote = normalize_text(q["evidence_quote"])
                # Use all occurrences. A chunk must cover a complete evidence span.
                offsets = [m.start() for m in re.finditer(re.escape(quote), doc["text"])]
                q["evidence_status"] = "quote_not_found" if not offsets else "span_crosses_chunks"
                q["gold_chunk_ids"] = [c["chunk_id"] for c in by_doc[doc["document_id"]]
                                       if any(c["char_start"] <= start and c["char_end"] >= start + len(quote)
                                              for start in offsets)]
                if q["gold_chunk_ids"]:
                    q["evidence_status"] = "mapped"
        elif q.get("evidence_quote") or q.get("source_document_id"):
            raise ValueError(f"Unanswerable question has positive evidence: {q['id']}")
        result["questions"].append(q)
    overlap = split_docs["development"] & split_docs["test"]
    if overlap:
        raise ValueError(f"Development/test article overlap: {sorted(overlap)}")
    result["mapping_summary"] = {
        "total": len(result["questions"]),
        "by_split": dict(Counter(q["split"] for q in result["questions"])),
        "answerable": sum(q["answerable"] for q in result["questions"]),
        "mapped": sum(q["evidence_status"] == "mapped" for q in result["questions"]),
        "evidence_status_counts": dict(Counter(q["evidence_status"] for q in result["questions"])),
        "development_test_article_overlap": sorted(overlap),
        "human_review_complete": False,
    }
    return result


def build(source: Path, output: Path, settings: Settings, questions: Path | None = None):
    source = source.resolve()
    rows = read_jsonl(source)
    documents, rejected, quality = prepare_documents(rows, settings)
    chunks = make_chunks(documents, settings)
    raw = output / "data/raw/wikipedia_source.jsonl"
    raw.parent.mkdir(parents=True, exist_ok=True)
    if source != raw.resolve():
        shutil.copyfile(source, raw)
    write_jsonl(output / "data/processed/documents.jsonl", documents)
    write_jsonl(output / "data/processed/chunks.jsonl", chunks)
    write_jsonl(output / "data/processed/rejected_articles.jsonl", rejected)
    report = {
        "pipeline_version": VERSION, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"local_input_name": source.name, "sha256": sha256(raw), "bytes": raw.stat().st_size,
                   "dataset_family": "wikimedia/wikipedia", "reported_subset": "20231101.en",
                   "subset_verification": "Reported in original project materials; local JSONL has no dump revision metadata.",
                   "dataset_card": "https://huggingface.co/datasets/wikimedia/wikipedia"},
        "settings": asdict(settings), "input_articles": len(rows),
        "retained_articles": len(documents), "rejected_articles": len(rejected),
        "rejection_reason_counts": dict(Counter(reason for r in rejected for reason in r["reasons"])),
        "quality_flags": quality, "raw_article_words": summary([len(WORD.findall(r["text"])) for r in rows]),
        "retained_article_words": summary([d["word_count"] for d in documents]),
        "chunks": len(chunks), "chunk_words": summary([c["word_count"] for c in chunks]),
        "duplicate_chunk_texts": len(chunks) - len({c["text"] for c in chunks}),
        "source_title_initials": dict(Counter(d["title"][0].upper() for d in documents)),
        "limits": ["The 200-row local source is an early general-topic slice, not a random sample or an astronomy collection.",
                   "max_articles is a cap, not an instruction to fabricate or download missing articles.",
                   "Chunk sizes use whitespace words, not model tokenizer tokens. Generator budgets need separate verification.",
                   "No stemming, stop-word removal, lowercasing or automatic factual correction of source text.",
                   "End matter is trimmed only at a standalone References, External links, Further reading or Bibliography heading after a substantial body. Other lists and extraction artifacts can remain.",
                   "Only preprocessing and a separate lexical retrieval baseline are implemented here. Dense RAG is a later stage."],
    }
    artifacts = ["data/raw/wikipedia_source.jsonl", "data/processed/documents.jsonl",
                 "data/processed/chunks.jsonl", "data/processed/rejected_articles.jsonl"]
    if questions is not None:
        question_snapshot = output / "evaluation/wikipedia_questions_draft.json"
        question_snapshot.parent.mkdir(parents=True, exist_ok=True)
        if questions.resolve() != question_snapshot.resolve():
            shutil.copyfile(questions, question_snapshot)
        mapped = map_questions(json.loads(question_snapshot.read_text(encoding="utf-8")), documents, chunks)
        write_json(output / "evaluation/questions_mapped.json", mapped)
        report["evaluation"] = mapped["mapping_summary"]
        artifacts.append("evaluation/questions_mapped.json")
        artifacts.append("evaluation/wikipedia_questions_draft.json")
    write_json(output / "reports/dataset_report.json", report)
    artifacts.append("reports/dataset_report.json")
    manifest = {"pipeline_version": VERSION, "settings": asdict(settings),
                "raw_source_sha256": sha256(raw),
                "files": {name: sha256(output / name) for name in artifacts}}
    write_json(output / "data/manifest.json", manifest)
    return report


def verify(output: Path):
    manifest = json.loads((output / "data/manifest.json").read_text(encoding="utf-8"))
    checked = []
    for name, expected in manifest["files"].items():
        path = (output / name).resolve()
        if not path.is_relative_to(output.resolve()):
            raise ValueError(f"Manifest path escapes output directory: {name}")
        if sha256(path) != expected:
            raise ValueError(f"Artifact checksum mismatch: {name}")
        checked.append(name)
    settings = Settings(**manifest["settings"])
    rows = read_jsonl(output / "data/raw/wikipedia_source.jsonl")
    expected_docs, expected_rejected, _ = prepare_documents(rows, settings)
    documents = read_jsonl(output / "data/processed/documents.jsonl")
    chunks = read_jsonl(output / "data/processed/chunks.jsonl")
    if documents != expected_docs or chunks != make_chunks(documents, settings):
        raise ValueError("Processed data differs from deterministic rebuild")
    rejected_path = output / "data/processed/rejected_articles.jsonl"
    rejected = read_jsonl(rejected_path) if rejected_path.stat().st_size else []
    if rejected != expected_rejected:
        raise ValueError("Rejection log differs from deterministic rebuild")
    if len({c["chunk_id"] for c in chunks}) != len(chunks):
        raise ValueError("Duplicate chunk ids")
    draft = output / "evaluation/wikipedia_questions_draft.json"
    mapped_path = output / "evaluation/questions_mapped.json"
    if "evaluation/questions_mapped.json" in manifest["files"]:
        expected_mapping = map_questions(json.loads(draft.read_text(encoding="utf-8")), documents, chunks)
        if expected_mapping != json.loads(mapped_path.read_text(encoding="utf-8")):
            raise ValueError("Evidence mapping differs from deterministic reconstruction")
    result = {"status": "passed", "files_checked": checked, "deterministic_rebuild": True,
              "documents": len(documents), "chunks": len(chunks),
              "network_needed": False, "scope": "Dataset preparation and artifact integrity only"}
    write_json(output / "reports/verification.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build", help="Inspect, clean, chunk and map draft evaluation evidence")
    build_parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    build_parser.add_argument("--output-dir", type=Path, default=ROOT)
    build_parser.add_argument("--questions", type=Path, default=ROOT / "evaluation/wikipedia_questions_draft.json")
    build_parser.add_argument("--skip-evaluation", action="store_true",
                              help="Prepare documents/passages without a local question file (source-only checkout)")
    build_parser.add_argument("--min-words", type=int, default=80)
    build_parser.add_argument("--chunk-words", type=int, default=200)
    build_parser.add_argument("--overlap-words", type=int, default=40)
    build_parser.add_argument("--max-articles", type=int, default=500)
    verify_parser = sub.add_parser("verify", help="Check hashes and deterministic reconstruction")
    verify_parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            if not args.skip_evaluation and not args.questions.is_file():
                parser.error(f"Question file does not exist: {args.questions}")
            report = build(args.source, args.output_dir, Settings(args.min_words, args.chunk_words,
                           args.overlap_words, args.max_articles), None if args.skip_evaluation else args.questions)
            print(json.dumps({key: report[key] for key in ("input_articles", "retained_articles", "rejected_articles", "chunks", "evaluation") if key in report}, indent=2))
        else:
            print(json.dumps(verify(args.output_dir), indent=2))
    except (ValueError, OSError, KeyError) as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
