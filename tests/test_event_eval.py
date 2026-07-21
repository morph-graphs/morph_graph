"""Unit tests for query-conditioned oracle event evaluation."""

from __future__ import annotations

import unittest
from dataclasses import replace

from morph_graph.event_eval.annotate import AnnotationSample, _filter_samples
from morph_graph.event_eval.gemini import build_agnostic_prompt, build_prompt
from morph_graph.event_eval.metrics import (
    boundary_f1,
    compare_annotation,
    compare_event,
    query_event_coverage,
)
from morph_graph.event_eval.query_taxonomy import classify_question
from morph_graph.event_eval.schema import OracleAnnotation, OracleEvent
from morph_graph.event_eval.select_pilot import select_records
from pathlib import Path


def _event(**overrides) -> OracleEvent:
    values = {
        "event_id": "e1",
        "start_sec": 8.0,
        "end_sec": 22.0,
        "granularity": "fine",
        "description": "The wearer intentionally searches the bedroom.",
        "categories": ("action", "location_or_scene"),
        "query_roles": ("answer_bearing",),
        "parent_event_ids": (),
    }
    values.update(overrides)
    return OracleEvent(**values)


def _annotation(event: OracleEvent | None = None) -> OracleAnnotation:
    return OracleAnnotation(
        sample_id="sample-1",
        video_id="video-1",
        video_path="/tmp/video-1.mp4",
        video_duration_sec=30.0,
        query="What did the wearer search?",
        events=(event or _event(),),
        model="gemini-test",
        prompt_version="test",
        video_fps=1.0,
        media_resolution="medium",
        generated_at="2026-07-17T00:00:00+00:00",
    )


class SchemaTests(unittest.TestCase):
    def test_valid_annotation(self) -> None:
        self.assertEqual(_annotation().validate(), [])

    def test_rejects_incorrrect_category_and_missing_role(self) -> None:
        event = _event(categories=("visible_object",), query_roles=())
        errors = _annotation(event).validate()
        self.assertTrue(any("unknown categories" in error for error in errors))
        self.assertTrue(any("needs a query role" in error for error in errors))

    def test_rejects_unknown_parent(self) -> None:
        errors = _annotation(_event(parent_event_ids=("missing",))).validate()
        self.assertTrue(any("unknown parent_event_ids" in error for error in errors))

    def test_parent_must_temporally_contain_child(self) -> None:
        parent = _event(
            event_id="parent",
            start_sec=5.0,
            end_sec=20.0,
            granularity="coarse",
            query_roles=("anchor",),
        )
        child = _event(
            event_id="child",
            start_sec=4.0,
            end_sec=10.0,
            parent_event_ids=("parent",),
        )
        errors = replace(_annotation(), events=(parent, child)).validate()
        self.assertTrue(any("is not contained by parent" in error for error in errors))


class MetricTests(unittest.TestCase):
    def test_fragmentation_and_contamination(self) -> None:
        result = compare_event(_event(), [[0, 10], [10, 20], [20, 30]])
        self.assertEqual(result["fragmentation"], 3)
        self.assertEqual(result["over_segmentation"], 2)
        self.assertEqual(result["covered_sec"], 14.0)
        self.assertEqual(result["contamination_sec"], 16.0)
        self.assertAlmostEqual(result["contamination_ratio"], 16 / 30, places=5)
        self.assertEqual(result["start_boundary_error_sec"], 8.0)
        self.assertEqual(result["end_boundary_error_sec"], 8.0)

    def test_boundary_matching_is_one_to_one(self) -> None:
        result = boundary_f1([9, 10], [10], tolerance_sec=2)
        self.assertEqual(result["matches"], 1)
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall"], 1.0)

    def test_annotation_comparison(self) -> None:
        result = compare_annotation(
            _annotation(),
            {
                "events_sec": [[0, 10], [10, 20], [20, 30]],
                "boundaries_sec": [10, 20],
            },
            tolerances_sec=(2,),
        )
        self.assertEqual(result["n_oracle_events"], 1)
        self.assertEqual(result["n_tsm_events"], 3)
        self.assertEqual(result["boundary_f1"]["2s"]["matches"], 2)
        self.assertEqual(result["summary"]["mean_over_segmentation"], 2.0)
        # One oracle event → no TSM segment merges multiple oracles.
        self.assertEqual(result["summary"]["mean_under_segmentation"], 0.0)

    def test_under_segmentation_when_tsm_merges_oracles(self) -> None:
        first = _event(event_id="e1", start_sec=5.0, end_sec=12.0)
        second = _event(event_id="e2", start_sec=14.0, end_sec=20.0)
        annotation = replace(_annotation(), events=(first, second))
        result = compare_annotation(
            annotation,
            {
                "events_sec": [[0, 30]],
                "boundaries_sec": [],
            },
            tolerances_sec=(2,),
        )
        self.assertEqual(result["summary"]["mean_over_segmentation"], 0.0)
        self.assertEqual(result["summary"]["mean_under_segmentation"], 1.0)
        self.assertEqual(result["tsm_under_segmentation"][0]["under_segmentation"], 1)

    def test_query_event_coverage_is_for_retrieved_spans(self) -> None:
        result = query_event_coverage(_event(), [[0, 10], [20, 30]])
        self.assertEqual(result["covered_sec"], 4.0)
        self.assertAlmostEqual(result["coverage_ratio"], 4 / 14, places=5)


class PromptTests(unittest.TestCase):
    def test_prompt_rejects_object_sighting_as_event(self) -> None:
        prompt = build_prompt("How many beds were visible?", 60.0)
        self.assertIn("Mere visibility of an object is NOT an event", prompt)
        self.assertIn("minimal set of meaningful video events", prompt)

    def test_agnostic_prompt_has_no_query_roles(self) -> None:
        prompt = build_agnostic_prompt(60.0)
        self.assertIn("query-agnostic", prompt.lower())
        self.assertIn("Do not include query roles", prompt)
        self.assertNotIn("answer_bearing", prompt)

    def test_agnostic_prompt_discourages_micro_tiling_and_fake_hierarchy(self) -> None:
        prompt = build_agnostic_prompt(600.0)
        self.assertIn("Compact partition", prompt)
        self.assertIn("near-duplicate", prompt)
        self.assertIn("NOT duration tags", prompt)
        self.assertIn("does **not** require any children", prompt)
        self.assertIn("wrap the whole video in one coarse parent", prompt)


class AgnosticSchemaTests(unittest.TestCase):
    def test_agnostic_allows_empty_roles(self) -> None:
        event = _event(query_roles=())
        annotation = replace(
            _annotation(event),
            annotation_mode="query_agnostic",
            query="",
        )
        self.assertEqual(annotation.validate(), [])

    def test_agnostic_rejects_nonempty_query(self) -> None:
        event = _event(query_roles=())
        annotation = replace(
            _annotation(event),
            annotation_mode="query_agnostic",
            query="should be empty",
        )
        errors = annotation.validate()
        self.assertTrue(any("leave query empty" in error for error in errors))


class PilotSelectionTests(unittest.TestCase):
    def test_temporal_locator_is_identification_not_ordering(self) -> None:
        taxonomy = classify_question(
            "After I left the grassy trail, what was the first structure I encountered?"
        )
        self.assertEqual(taxonomy.evidence_scope, "single_target")
        self.assertIn("temporal_constraint", taxonomy.link_bases)
        self.assertEqual(taxonomy.answer_operation, "identify")

    def test_selects_distinct_profiles_and_videos(self) -> None:
        records = [
            {
                "id": 1,
                "video": "a.mp4",
                "question": "After leaving the trail, what structure did I encounter?",
                "category": "A",
            },
            {
                "id": 2,
                "video": "b.mp4",
                "question": "How was the same red car different the second time?",
                "category": "B",
            },
            {
                "id": 3,
                "video": "c.mp4",
                "question": "What was the total duration?",
                "category": "C",
            },
        ]
        selected = select_records(
            records,
            count=2,
            profiles=("temporal_locator", "shared_entity_comparison"),
        )
        self.assertEqual([record["video"] for record in selected], ["a.mp4", "b.mp4"])


class AnnotateFilterTests(unittest.TestCase):
    def test_filter_by_sample_id_and_video_id(self) -> None:
        samples = [
            AnnotationSample("510", Path("/data/val/b3fba3c9b747e519.mp4"), "q1"),
            AnnotationSample("73", Path("/data/val/1cc66eed89b44527.mp4"), "q2"),
            AnnotationSample("511", Path("/data/val/b3fba3c9b747e519.mp4"), "q3"),
        ]
        by_sample = _filter_samples(samples, sample_ids=["510"])
        self.assertEqual([s.sample_id for s in by_sample], ["510"])

        by_video = _filter_samples(
            samples, video_ids=["b3fba3c9b747e519.mp4"]
        )
        self.assertEqual([s.sample_id for s in by_video], ["510", "511"])

        both = _filter_samples(
            samples, sample_ids=["511", "73"], video_ids=["b3fba3c9b747e519"]
        )
        self.assertEqual([s.sample_id for s in both], ["511"])


if __name__ == "__main__":
    unittest.main()
