"""Regression checks for meaningful lexical retrieval and saved corpus integrity."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval_baseline import BM25Index, build_index, evaluate, load_index, search


def chunk(identifier, text, document="wiki:1"):
    return {"chunk_id": identifier, "document_id": document, "title": identifier, "url": "https://example.test", "text": text}


class RetrievalBaselineTests(unittest.TestCase):
    def test_rare_terms_rank_relevant_evidence_first(self):
        index = BM25Index([
            chunk("a", "Astronomers study celestial objects and distant galaxies."),
            chunk("b", "Albedo is the fraction of incoming solar radiation reflected by a surface."),
            chunk("c", "Radiation and objects can be measured."),
        ])
        results = search(index, "What fraction of incoming solar radiation is reflected?", 3)
        self.assertEqual(results[0]["chunk_id"], "b")
        self.assertGreater(results[0]["score"], results[1]["score"])

    def test_length_normalization_and_deterministic_ties(self):
        index = BM25Index([chunk("z", "orbit planets"), chunk("a", "orbit planets"), chunk("long", "orbit " + "other " * 50)])
        self.assertEqual([row["chunk_id"] for row in search(index, "orbit")], ["a", "z", "long"])

    def test_zero_matching_terms_returns_no_invented_results(self):
        index = BM25Index([chunk("a", "astronomy galaxies")])
        self.assertEqual(search(index, "unicorns"), [])
        self.assertEqual(search(index, "what is the"), [])

    def test_duplicate_chunk_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate chunk_id"):
            BM25Index([chunk("a", "first"), chunk("a", "second")])

    def test_saved_roundtrip_and_corpus_drift_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            source, artifact = Path(directory) / "chunks.jsonl", Path(directory) / "bm25.json"
            source.write_text("\n".join(json.dumps(row) for row in [chunk("a", "orbit planet"), chunk("b", "surface reflection")]), encoding="utf-8")
            built = build_index(source, artifact)
            loaded = load_index(artifact)
            self.assertEqual(search(built, "reflection"), search(loaded, "reflection"))
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            payload["chunks"][0]["text"] = "tampered corpus"
            artifact.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                load_index(artifact)

    def test_metric_denominator_excludes_unmapped_and_unanswerable(self):
        index = BM25Index([chunk("a", "reflected solar radiation"), chunk("b", "distant galaxies", "wiki:2")])
        report = evaluate(index, {"status": "draft_requires_human_review", "questions": [
            {"id": "q1", "split": "test", "question": "reflected radiation", "answerable": True, "evidence_status": "mapped", "gold_chunk_ids": ["a"], "document_id": "wiki:1"},
            {"id": "q2", "split": "test", "question": "galaxies", "answerable": True, "evidence_status": "unmapped", "gold_chunk_ids": []},
            {"id": "q3", "split": "development", "question": "unknown unicorn", "answerable": False, "gold_chunk_ids": []},
        ]})
        self.assertEqual(report["overall"]["mapped_answerable_denominator"], 1)
        self.assertEqual(report["overall"]["evidence_hit_at_1"], {"hits": 1, "denominator": 1, "rate": 1.0})
        self.assertEqual(report["by_split"]["development"]["evidence_hit_at_5"]["rate"], None)
        self.assertEqual(report["unanswerable_retrieval_scores"][0]["top_score"], None)

    def test_evidence_corpus_drift_is_rejected(self):
        index = BM25Index([chunk("a", "astronomy")])
        with self.assertRaisesRegex(ValueError, "missing chunks"):
            evaluate(index, {"questions": [{"id": "q1", "question": "astronomy", "answerable": True, "evidence_status": "mapped", "gold_chunk_ids": ["old-chunk"]}]})


if __name__ == "__main__":
    unittest.main()
