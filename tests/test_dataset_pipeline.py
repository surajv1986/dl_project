from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from dataset_pipeline import (Settings, build, make_chunks, map_questions,
                              main, normalize_text, prepare_documents, remove_end_matter, verify)


def article(article_id="1", text=None, title="Astronomy"):
    return {"id": article_id, "title": title, "url": f"https://en.wikipedia.org/wiki/{title}",
            "text": text or " ".join(f"word{i}" for i in range(450))}


class PipelineTests(unittest.TestCase):
    def test_normalization_preserves_scientific_meaning(self):
        self.assertEqual(normalize_text("  H₂O\tand Δv = 3.5 m/s\r\n\r\nCASE  \x00"), "H₂O and Δv = 3.5 m/s\n\nCASE")

    def test_filters_and_conflicting_ids(self):
        rows = [article(), article("2", title="Austin (disambiguation)"), article("3", "Short."), article("4")]
        docs, rejects, _ = prepare_documents(rows, Settings())
        self.assertEqual(len(docs), 1)
        self.assertEqual(len(rejects), 3)
        self.assertIn("duplicate_normalized_text", rejects[-1]["reasons"])
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            prepare_documents([article(), article(text="different " * 90)], Settings())

    def test_end_matter_heading_is_conservative(self):
        body = "The References section is mentioned in this sentence. " + "body " * 90
        text = body + "\n\nReferences\nCitation list\nExternal links\nExample"
        cleaned, heading, removed = remove_end_matter(text)
        self.assertEqual(cleaned, body.rstrip())
        self.assertEqual(heading, "References")
        self.assertGreater(removed, 0)
        self.assertEqual(remove_end_matter("References\n" + "body " * 90)[0], "References\n" + "body " * 90)

    def test_chunks_cover_every_character_span_of_words(self):
        docs, _, _ = prepare_documents([article()], Settings())
        chunks = make_chunks(docs, Settings())
        self.assertEqual([c["word_count"] for c in chunks], [200, 200, 130])
        self.assertEqual(chunks[0]["word_end"] - chunks[1]["word_start"], 40)
        self.assertEqual(chunks[-1]["char_end"], len(docs[0]["text"]))
        for chunk in chunks:
            self.assertEqual(chunk["text"], docs[0]["text"][chunk["char_start"]:chunk["char_end"]])
        self.assertEqual(chunks, make_chunks(docs, Settings()))
        self.assertNotEqual(chunks[0]["chunk_id"], make_chunks(docs, Settings(chunk_words=201))[0]["chunk_id"])

    def test_full_evidence_span_and_split_leakage(self):
        docs, _, _ = prepare_documents([article()], Settings())
        chunks = make_chunks(docs, Settings())
        question = {"id": "Q1", "split": "development", "answerable": True,
                    "source_document_id": "1", "question": "Which word?",
                    "reference_answers": ["word170"], "evidence_quote": "word169 word170 word171"}
        mapped = map_questions({"questions": [question]}, docs, chunks)
        self.assertEqual(len(mapped["questions"][0]["gold_chunk_ids"]), 2)
        other = dict(question, id="Q2", split="test")
        with self.assertRaisesRegex(ValueError, "overlap"):
            map_questions({"questions": [question, other]}, docs, chunks)

    def test_portable_roundtrip_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            source.write_text(json.dumps(article()) + "\n", encoding="utf-8")
            output = root / "output"
            build(source, output, Settings())
            self.assertEqual(verify(output)["status"], "passed")
            with (output / "data/processed/chunks.jsonl").open("a", encoding="utf-8") as stream:
                stream.write("{}\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify(output)

    def test_empty_source_and_invalid_window_fail(self):
        with self.assertRaises(ValueError):
            make_chunks([], Settings(chunk_words=40, overlap_words=40))
        with self.assertRaisesRegex(ValueError, "required"):
            prepare_documents([{"id": "1"}], Settings())

    def test_source_only_cli_needs_no_evaluation_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            source.write_text(json.dumps(article()) + "\n", encoding="utf-8")
            output = root / "output"
            with redirect_stdout(io.StringIO()):
                result = main(["build", "--source", str(source), "--output-dir", str(output),
                               "--questions", str(root / "not_distributed.json"), "--skip-evaluation"])
            self.assertEqual(result, 0)
            self.assertFalse((output / "evaluation/questions_mapped.json").exists())
            self.assertEqual(verify(output)["status"], "passed")


if __name__ == "__main__":
    unittest.main()
