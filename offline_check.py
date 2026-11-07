"""Rebuild the dataset and reload/query the lexical index with Python networking blocked."""

from __future__ import annotations

import json
from pathlib import Path
import platform
import socket
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parent


def main():
    attempts = []

    def blocked(*args, **kwargs):
        attempts.append("network_call_blocked")
        raise RuntimeError("Networking is disabled for this offline verification")

    # Block network connections/DNS before importing either pipeline module.
    # Local filesystem reads continue to work. This is a Python socket-level test,
    # not an OS firewall test or a test of future model-based RAG components.
    socket.create_connection = blocked
    socket.getaddrinfo = blocked
    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked

    from dataset_pipeline import Settings, build, read_jsonl, sha256, verify, write_json
    from retrieval_baseline import build_index, corpus_digest, evaluate, load_index, search

    start = time.perf_counter()
    shipped_verification = verify(ROOT)
    raw = ROOT / "data/raw/wikipedia_source.jsonl"
    question_file = ROOT / "evaluation/wikipedia_questions_draft.json"
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="wikipedia_offline_") as temporary:
        out = Path(temporary)
        rebuilt = build(raw, out, Settings(**manifest["settings"]), question_file)
        verify(out)
        for name in ("documents.jsonl", "chunks.jsonl", "rejected_articles.jsonl"):
            if sha256(ROOT / "data/processed" / name) != sha256(out / "data/processed" / name):
                raise ValueError(f"Offline rebuild differs: {name}")
        if sha256(ROOT / "evaluation/questions_mapped.json") != sha256(out / "evaluation/questions_mapped.json"):
            raise ValueError("Offline evidence mapping differs")
        fresh = build_index(out / "data/processed/chunks.jsonl", out / "bm25.json")
        loaded = load_index(out / "bm25.json")
        shipped = load_index(ROOT / "data/index/bm25.json")
        expected_digest = corpus_digest(read_jsonl(ROOT / "data/processed/chunks.jsonl"))
        if not loaded.corpus_sha256 == shipped.corpus_sha256 == fresh.corpus_sha256 == expected_digest:
            raise ValueError("Saved index and current corpus differ")
        if (shipped.k1, shipped.b) != (loaded.k1, loaded.b):
            raise ValueError("Saved index uses different BM25 parameters")
        question = "What was the name of the Apollo 11 lunar module?"
        matches = search(loaded, question, 3)
        if matches != search(shipped, question, 3):
            raise ValueError("Saved index query differs from the rebuilt index")
        if not matches or "Eagle" not in " ".join(match["text"] for match in matches):
            raise ValueError("Demonstration query did not retrieve expected evidence")
        metrics = evaluate(loaded, json.loads((out / "evaluation/questions_mapped.json").read_text(encoding="utf-8")))
        shipped_metrics = evaluate(shipped, json.loads((ROOT / "evaluation/questions_mapped.json").read_text(encoding="utf-8")))
        if metrics["questions"] != shipped_metrics["questions"]:
            raise ValueError("Saved index evaluation differs from the rebuilt index")
        if attempts:
            raise ValueError("Pipeline attempted a network connection")
        result = {
            "status": "passed", "python_version": platform.python_version(),
            "platform": platform.platform(), "network_block": "Python socket connection and DNS methods replaced before module imports",
            "network_attempts": len(attempts), "source_sha256": sha256(raw),
            "deterministic_processed_artifacts": True, "evidence_mapping_identical": True,
            "index_roundtrip_and_corpus_digest": True, "documents": rebuilt["retained_articles"],
            "chunks": rebuilt["chunks"], "draft_evaluation_questions": metrics["overall"]["question_count"],
            "demonstration": {"question": question, "top_titles": [m["title"] for m in matches],
                              "top_chunk_id": matches[0]["chunk_id"]},
            "elapsed_seconds": round(time.perf_counter() - start, 3),
            "scope": "Dataset rebuild, artifact verification, lexical index save/reload, retrieval and draft evidence-hit calculation",
            "not_verified": ["Dense embeddings", "Generative model inference", "Full end-to-end RAG", "OS firewall isolation"],
        }
    write_json(ROOT / "reports/offline_verification.json", result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
