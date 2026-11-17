from io import BytesIO
import unittest

from docx import Document

from rag_engine import DocumentIndex, chunk_text, extract_text


class ChunkTextTests(unittest.TestCase):
    def test_chunks_overlap_without_losing_words(self):
        chunks = chunk_text("one two three four five", chunk_size=3, overlap=1)
        self.assertEqual(chunks, ["one two three", "three four five"])

    def test_rejects_invalid_overlap(self):
        with self.assertRaises(ValueError):
            chunk_text("some words", chunk_size=3, overlap=3)


class DocumentIndexTests(unittest.TestCase):
    def test_bm25_returns_most_relevant_source_first(self):
        index = DocumentIndex()
        index.add_document("garden.txt", "Tomatoes need sunlight and moist soil.")
        index.add_document("space.txt", "A rocket launches satellites into orbit.")

        results = index.search("tomatoes sunlight")

        self.assertEqual(results[0].source, "garden.txt")
        self.assertGreater(results[0].score, 0)

    def test_replacing_document_does_not_keep_stale_chunks(self):
        index = DocumentIndex()
        index.add_document("notes.txt", "Old material about astronomy.")
        index.add_document("notes.txt", "New material about botany.")

        self.assertEqual(index.sources, ["notes.txt"])
        self.assertEqual(index.search("astronomy"), [])
        self.assertEqual(index.search("botany")[0].source, "notes.txt")

    def test_vector_retrieval_uses_cosine_similarity(self):
        def embed(texts):
            return [
                [1.0, 0.0]
                if "cats" in text.lower() or "felines" in text.lower()
                else [0.0, 1.0]
                for text in texts
            ]

        index = DocumentIndex(embedder=embed)
        index.add_document("cats.txt", "Cats are curious pets.")
        index.add_document("ocean.txt", "Ocean waves move along the shore.")

        self.assertEqual(index.search("felines")[0].source, "cats.txt")

    def test_failed_embedding_does_not_replace_existing_document(self):
        index = DocumentIndex(embedder=lambda texts: [[1.0, 0.0] for _ in texts])
        index.add_document("notes.txt", "Original useful information.")

        def fail(_texts):
            raise RuntimeError("offline")

        with self.assertRaises(RuntimeError):
            index.set_embedder(fail)

        self.assertEqual(index.search("original")[0].source, "notes.txt")


class ExtractionTests(unittest.TestCase):
    def test_extracts_utf8_text(self):
        self.assertEqual(extract_text("notes.txt", b"Useful notes"), "Useful notes")

    def test_rejects_unsupported_file_type(self):
        with self.assertRaisesRegex(ValueError, "Unsupported file type"):
            extract_text("image.png", b"not a document")

    def test_rejects_empty_text(self):
        with self.assertRaisesRegex(ValueError, "No readable text"):
            extract_text("empty.md", b"  \n")

    def test_extracts_docx_paragraphs_and_tables(self):
        buffer = BytesIO()
        document = Document()
        document.add_paragraph("Project summary")
        table = document.add_table(rows=1, cols=2)
        table.cell(0, 0).text = "Owner"
        table.cell(0, 1).text = "CogniSync"
        document.save(buffer)

        text = extract_text("brief.docx", buffer.getvalue())

        self.assertIn("Project summary", text)
        self.assertIn("Owner | CogniSync", text)


if __name__ == "__main__":
    unittest.main()
