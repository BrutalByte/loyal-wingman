"""
Unit tests for loyal_wingman.cli's lesson storage and retrieval logic.

Scope: _cosine, _load_lessons, _save_lessons, _backfill_embeddings,
_select_lessons, _format_lessons_block. Deliberately does not test the
CLI subcommand handlers (cmd_run/cmd_teach/cmd_lessons) or LM Studio
process management (_ensure_server/_ensure_model/_ensure_embedding_model)
-- those need real subprocess/HTTP integration, not unit coverage.

No network or LM Studio server required: _embed is mocked throughout via
unittest.mock.patch, and LESSONS_PATH is redirected to a temp file per
test so nothing here touches ~/.loyal-wingman/lessons.jsonl.

Run with:  python -m unittest discover -s tests
       or: pytest tests/
"""
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from loyal_wingman import cli


class TestCosine(unittest.TestCase):
    def test_identical_vectors(self):
        self.assertAlmostEqual(cli._cosine([1, 0, 0], [1, 0, 0]), 1.0)

    def test_orthogonal_vectors(self):
        self.assertAlmostEqual(cli._cosine([1, 0], [0, 1]), 0.0)

    def test_opposite_vectors(self):
        self.assertAlmostEqual(cli._cosine([1, 0], [-1, 0]), -1.0)

    def test_zero_vector_returns_zero_not_crash(self):
        self.assertEqual(cli._cosine([0, 0, 0], [1, 2, 3]), 0.0)
        self.assertEqual(cli._cosine([1, 2, 3], [0, 0, 0]), 0.0)
        self.assertEqual(cli._cosine([0, 0], [0, 0]), 0.0)

    def test_scale_invariant(self):
        # Cosine similarity depends only on direction, not magnitude.
        self.assertAlmostEqual(cli._cosine([1, 1], [2, 2]), 1.0)

    def test_known_angle(self):
        # 45 degrees apart -> cos(45 deg) == sqrt(2)/2
        self.assertAlmostEqual(cli._cosine([1, 0], [1, 1]), math.sqrt(2) / 2, places=6)


class LessonsFileTestCase(unittest.TestCase):
    """Redirects cli.LESSONS_PATH to an isolated temp file for the test."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.lessons_path = Path(self._tmpdir.name) / "lessons.jsonl"
        self._patcher = patch.object(cli, "LESSONS_PATH", self.lessons_path)
        self._patcher.start()
        self.addCleanup(self._patcher.stop)
        self.addCleanup(self._tmpdir.cleanup)

    def _write_lessons(self, lessons):
        self.lessons_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.lessons_path, "w", encoding="utf-8") as f:
            for lesson in lessons:
                f.write(json.dumps(lesson) + "\n")


class TestLoadSaveLessons(LessonsFileTestCase):
    def test_load_nonexistent_file_returns_empty_list(self):
        self.assertEqual(cli._load_lessons(), [])

    def test_load_skips_blank_lines(self):
        self.lessons_path.parent.mkdir(parents=True, exist_ok=True)
        self.lessons_path.write_text(
            '{"category": "x", "issue": "a", "fix": "b"}\n\n\n', encoding="utf-8"
        )
        self.assertEqual(len(cli._load_lessons()), 1)

    def test_load_skips_malformed_json_line(self):
        self.lessons_path.parent.mkdir(parents=True, exist_ok=True)
        self.lessons_path.write_text(
            '{"category": "x", "issue": "a", "fix": "b"}\n'
            "not valid json\n"
            '{"category": "y", "issue": "c", "fix": "d"}\n',
            encoding="utf-8",
        )
        lessons = cli._load_lessons()
        self.assertEqual([l["issue"] for l in lessons], ["a", "c"])

    def test_save_then_load_round_trips(self):
        lessons = [
            {"category": "general", "issue": "a", "fix": "b", "embedding": [0.1, 0.2]},
            {"category": "x", "issue": "c", "fix": "d"},
        ]
        cli._save_lessons(lessons)
        self.assertEqual(cli._load_lessons(), lessons)

    def test_save_creates_parent_directory(self):
        nested = Path(self._tmpdir.name) / "nested" / "dir" / "lessons.jsonl"
        with patch.object(cli, "LESSONS_PATH", nested):
            cli._save_lessons([{"category": "general", "issue": "a", "fix": "b"}])
            self.assertTrue(nested.exists())


class TestBackfillEmbeddings(unittest.TestCase):
    def test_embeds_lesson_missing_embedding(self):
        lessons = [{"category": "g", "issue": "hello", "fix": "world"}]
        with patch.object(cli, "_embed", return_value=[1.0, 2.0]) as mock_embed:
            changed = cli._backfill_embeddings(lessons, "http://fake", "fake-model")
        self.assertTrue(changed)
        self.assertEqual(lessons[0]["embedding"], [1.0, 2.0])
        mock_embed.assert_called_once_with("http://fake", "fake-model", "hello")

    def test_does_not_reembed_existing_embedding(self):
        lessons = [{"category": "g", "issue": "hello", "fix": "world", "embedding": [9.0]}]
        with patch.object(cli, "_embed", return_value=[1.0, 2.0]) as mock_embed:
            changed = cli._backfill_embeddings(lessons, "http://fake", "fake-model")
        self.assertFalse(changed)
        self.assertEqual(lessons[0]["embedding"], [9.0])
        mock_embed.assert_not_called()

    def test_embed_failure_leaves_lesson_unembedded(self):
        lessons = [{"category": "g", "issue": "hello", "fix": "world"}]
        with patch.object(cli, "_embed", return_value=None):
            changed = cli._backfill_embeddings(lessons, "http://fake", "fake-model")
        self.assertFalse(changed)
        self.assertNotIn("embedding", lessons[0])

    def test_mixed_batch_only_embeds_missing_ones(self):
        lessons = [
            {"category": "g", "issue": "already has one", "fix": "x", "embedding": [9.0]},
            {"category": "g", "issue": "needs one", "fix": "y"},
        ]
        with patch.object(cli, "_embed", return_value=[1.0]) as mock_embed:
            changed = cli._backfill_embeddings(lessons, "http://fake", "fake-model")
        self.assertTrue(changed)
        mock_embed.assert_called_once_with("http://fake", "fake-model", "needs one")


class TestSelectLessons(LessonsFileTestCase):
    """The core retrieval/ranking logic."""

    def test_no_lessons_returns_empty(self):
        result = cli._select_lessons(
            "prompt", None, "http://fake", "", top_k=5, min_similarity=0.5
        )
        self.assertEqual(result, [])

    def test_no_embed_model_falls_back_to_recency(self):
        self._write_lessons([
            {"category": "general", "issue": "first", "fix": "a"},
            {"category": "general", "issue": "second", "fix": "b"},
            {"category": "general", "issue": "third", "fix": "c"},
        ])
        result = cli._select_lessons(
            "prompt", None, "http://fake", "", top_k=2, min_similarity=0.5
        )
        self.assertEqual([l["issue"] for l in result], ["second", "third"])

    def test_category_filter_keeps_matching_and_general_only(self):
        self._write_lessons([
            {"category": "changelog", "issue": "a", "fix": "x"},
            {"category": "docstring", "issue": "b", "fix": "y"},
            {"category": "general", "issue": "c", "fix": "z"},
        ])
        result = cli._select_lessons(
            "prompt", "changelog", "http://fake", "", top_k=5, min_similarity=0.5
        )
        self.assertEqual({l["issue"] for l in result}, {"a", "c"})

    def test_semantic_ranking_orders_most_similar_first(self):
        self._write_lessons([
            {"category": "general", "issue": "far", "fix": "x", "embedding": [0.0, 1.0]},
            {"category": "general", "issue": "near", "fix": "y", "embedding": [0.99, 0.14]},
            {"category": "general", "issue": "exact", "fix": "z", "embedding": [1.0, 0.0]},
        ])
        with patch.object(cli, "_embed", return_value=[1.0, 0.0]):
            result = cli._select_lessons(
                "prompt", None, "http://fake", "fake-model", top_k=5, min_similarity=0.0
            )
        self.assertEqual([l["issue"] for l in result], ["exact", "near", "far"])

    def test_similarity_floor_excludes_dissimilar_lessons(self):
        self._write_lessons([
            {"category": "general", "issue": "similar", "fix": "x", "embedding": [1.0, 0.0]},
            {"category": "general", "issue": "dissimilar", "fix": "y", "embedding": [0.0, 1.0]},
        ])
        with patch.object(cli, "_embed", return_value=[1.0, 0.0]):
            result = cli._select_lessons(
                "prompt", None, "http://fake", "fake-model", top_k=5, min_similarity=0.5
            )
        self.assertEqual([l["issue"] for l in result], ["similar"])

    def test_top_k_caps_results_even_if_all_pass_threshold(self):
        self._write_lessons([
            {"category": "general", "issue": f"lesson{i}", "fix": "x", "embedding": [1.0, 0.0]}
            for i in range(10)
        ])
        with patch.object(cli, "_embed", return_value=[1.0, 0.0]):
            result = cli._select_lessons(
                "prompt", None, "http://fake", "fake-model", top_k=3, min_similarity=0.0
            )
        self.assertEqual(len(result), 3)

    def test_query_embed_failure_falls_back_to_recency_ignoring_threshold(self):
        self._write_lessons([
            {"category": "general", "issue": "first", "fix": "a", "embedding": [1.0, 0.0]},
            {"category": "general", "issue": "second", "fix": "b", "embedding": [0.0, 1.0]},
        ])
        with patch.object(cli, "_embed", return_value=None):
            result = cli._select_lessons(
                "prompt", None, "http://fake", "fake-model", top_k=5, min_similarity=0.99
            )
        self.assertEqual([l["issue"] for l in result], ["first", "second"])

    def test_lesson_unembeddable_after_failed_backfill_is_skipped_not_crashed(self):
        self._write_lessons([{"category": "general", "issue": "no embedding possible", "fix": "x"}])

        def fake_embed(base_url, model, text):
            # Backfill call embeds the issue text and fails; query call embeds
            # the prompt and succeeds -- exercises both paths in one test.
            return None if text == "no embedding possible" else [1.0, 0.0]

        with patch.object(cli, "_embed", side_effect=fake_embed):
            result = cli._select_lessons(
                "prompt", None, "http://fake", "fake-model", top_k=5, min_similarity=0.0
            )
        self.assertEqual(result, [])

    def test_backfilled_embedding_persists_to_disk(self):
        self._write_lessons([{"category": "general", "issue": "needs embedding", "fix": "x"}])
        with patch.object(cli, "_embed", return_value=[1.0, 0.0]):
            cli._select_lessons(
                "prompt", None, "http://fake", "fake-model", top_k=5, min_similarity=0.0
            )
        # Reload from disk with no mock active -- confirms _save_lessons ran.
        reloaded = cli._load_lessons()
        self.assertEqual(reloaded[0]["embedding"], [1.0, 0.0])

    def test_backfill_embeds_lessons_outside_the_requested_category_too(self):
        # _backfill_embeddings runs on the full unfiltered set before the
        # category filter is applied, so the whole store stays embedded over
        # time regardless of which category happens to be queried first.
        self._write_lessons([
            {"category": "other-category", "issue": "unrelated", "fix": "x"},
        ])
        with patch.object(cli, "_embed", return_value=[1.0, 0.0]) as mock_embed:
            cli._select_lessons(
                "prompt", "changelog", "http://fake", "fake-model", top_k=5, min_similarity=0.0
            )
        mock_embed.assert_any_call("http://fake", "fake-model", "unrelated")
        reloaded = cli._load_lessons()
        self.assertIn("embedding", reloaded[0])

    def test_general_category_included_alongside_explicit_category_filter(self):
        self._write_lessons([
            {"category": "general", "issue": "general lesson", "fix": "x", "embedding": [1.0, 0.0]},
            {"category": "other", "issue": "other lesson", "fix": "y", "embedding": [1.0, 0.0]},
        ])
        with patch.object(cli, "_embed", return_value=[1.0, 0.0]):
            result = cli._select_lessons(
                "prompt", "specific-category", "http://fake", "fake-model", top_k=5, min_similarity=0.0
            )
        self.assertEqual([l["issue"] for l in result], ["general lesson"])


class TestFormatLessonsBlock(unittest.TestCase):
    def test_empty_list_returns_empty_string(self):
        self.assertEqual(cli._format_lessons_block([]), "")

    def test_formats_category_issue_and_fix(self):
        block = cli._format_lessons_block(
            [{"category": "changelog", "issue": "wrong format", "fix": "right format"}]
        )
        self.assertIn("[changelog]", block)
        self.assertIn("wrong format", block)
        self.assertIn("right format", block)
        self.assertIn("->", block)

    def test_missing_category_defaults_to_general(self):
        block = cli._format_lessons_block([{"issue": "a", "fix": "b"}])
        self.assertIn("[general]", block)

    def test_multiple_lessons_each_get_their_own_bullet(self):
        block = cli._format_lessons_block([
            {"category": "a", "issue": "issue1", "fix": "fix1"},
            {"category": "b", "issue": "issue2", "fix": "fix2"},
        ])
        self.assertEqual(block.count("- ["), 2)


if __name__ == "__main__":
    unittest.main()
