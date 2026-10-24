"""Unit tests for query-conditioned oracle event evaluation."""

from __future__ import annotations

import json
import unittest
from dataclasses import replace

from morph_graph.event_eval.annotate import AnnotationSample, _filter_samples, build_parser
from morph_graph.event_eval.gemini import build_agnostic_prompt, build_prompt
from morph_graph.event_eval.kimi import KimiAnnotationError, KimiOracleAnnotator
from morph_graph.event_eval.metrics import (
    boundary_f1,
    compare_annotation,
    compare_event,
    query_event_coverage,
)
from morph_graph.event_eval.query_taxonomy import classify_question
from morph_graph.event_eval.schema import (
    OracleAnnotation,
    OracleEvent,
    answers_match,
    gemini_response_schema,
    mcq_letter,
    normalize_answer,
)
from morph_graph.event_eval.select_pilot import select_records
from morph_graph.event_eval.verify import (
    build_verification_prompt,
    coverage_ratio,
    decide_verdict,
    parse_mcq_options,
    shuffled_options,
    verification_response_schema,
)
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

    def test_answer_check_match_and_mismatch(self) -> None:
        matched = replace(
            _annotation(), predicted_answer="The bedroom closet"
        ).with_answer_check("the bedroom closet.")
        self.assertTrue(matched.answer_match)
        self.assertEqual(matched.gt_answer, "the bedroom closet.")
        self.assertFalse(
            any("answer_check_failed" in warning for warning in matched.warnings)
        )

        mismatched = replace(
            _annotation(), predicted_answer="kitchen"
        ).with_answer_check("bedroom")
        self.assertFalse(mismatched.answer_match)
        self.assertTrue(
            any("answer_check_failed" in warning for warning in mismatched.warnings)
        )
        self.assertEqual(normalize_answer("A, B!"), "a b")
        self.assertTrue(answers_match("Yes.", "yes"))
        self.assertIsNone(answers_match(None, "yes"))

    def test_mcq_answer_check_compares_letters(self) -> None:
        options = "A. A stone wall B. A trash can C. A concrete bench D. A post"
        right = replace(
            _annotation(), options=options, predicted_answer="C. A concrete bench"
        ).with_answer_check("A concrete bench on the left.", mcq_answer="C")
        self.assertTrue(right.answer_match)
        self.assertEqual(right.gt_answer, "A concrete bench on the left.")

        wrong = replace(
            _annotation(), options=options, predicted_answer="(B) trash can"
        ).with_answer_check("A concrete bench on the left.", mcq_answer="C")
        self.assertFalse(wrong.answer_match)
        self.assertTrue(
            any("answer_check_failed" in warning for warning in wrong.warnings)
        )

        # A free-text answer that starts with the article "A" is not option A,
        # and an answer with no option letter counts as a mismatch.
        unparsed = replace(
            _annotation(), options=options, predicted_answer="A concrete bench"
        ).with_answer_check("A concrete bench on the left.", mcq_answer="C")
        self.assertFalse(unparsed.answer_match)
        missing = replace(
            _annotation(), options=options, predicted_answer=None
        ).with_answer_check("A concrete bench on the left.", mcq_answer="C")
        self.assertIsNone(missing.answer_match)
        self.assertEqual(mcq_letter("D"), "D")
        self.assertIsNone(mcq_letter("A fenced playground"))

    def test_options_round_trip(self) -> None:
        annotation = replace(_annotation(), options="A. x B. y")
        restored = OracleAnnotation.from_dict(json.loads(json.dumps(annotation.to_dict())))
        self.assertEqual(restored.options, "A. x B. y")

    def test_reads_pilot_answer_field_as_predicted_answer(self) -> None:
        raw = _annotation().to_dict()
        raw.pop("predicted_answer")
        raw["answer"] = "D. A wooden post"
        self.assertEqual(
            OracleAnnotation.from_dict(raw).predicted_answer, "D. A wooden post"
        )

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
    def test_prompt_asks_for_event_sufficient_answer(self) -> None:
        prompt = build_prompt("How many beds were visible?", 60.0)
        self.assertIn("visibility of a random object is not an event", prompt)
        self.assertIn("minimal set of meaningful video events", prompt)
        self.assertIn("predicted_answer", prompt)
        self.assertIn("derivable from the events you listed alone", prompt)
        self.assertNotIn("Multiple-choice options:", prompt)
        self.assertNotIn("option letter", prompt)

    def test_prompt_includes_mcq_options(self) -> None:
        prompt = build_prompt(
            "What did I see?",
            60.0,
            options="A. A tree B. A dog C. A car D. A plane",
        )
        self.assertIn("Multiple-choice options:", prompt)
        self.assertIn("A. A tree B. A dog C. A car D. A plane", prompt)
        self.assertIn("option letter", prompt)

    def test_conditioned_schema_requires_predicted_answer(self) -> None:
        conditioned = gemini_response_schema("query_conditioned")
        self.assertIn("predicted_answer", conditioned["properties"])
        self.assertIn("predicted_answer", conditioned["required"])
        self.assertNotIn("answer", conditioned["properties"])
        agnostic = gemini_response_schema("query_agnostic")
        self.assertNotIn("predicted_answer", agnostic["properties"])

    def test_agnostic_prompt_has_no_query_roles(self) -> None:
        prompt = build_agnostic_prompt(60.0)
        self.assertIn("query-agnostic", prompt.lower())
        self.assertIn("Do not include query roles", prompt)
        self.assertNotIn("answer_bearing", prompt)
        self.assertNotIn("predicted_answer", prompt)

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

    def test_expanded_fill_follows_operation_mix(self) -> None:
        records = [
            {
                "id": index,
                "video": f"id{index}.mp4",
                "question": "After leaving the trail, what structure did I encounter?",
                "category": "A",
            }
            for index in range(20)
        ] + [
            {
                "id": 100 + index,
                "video": f"cmp{index}.mp4",
                "question": "How was the same red car different the second time?",
                "category": "B",
            }
            for index in range(10)
        ]
        selected = select_records(
            records,
            count=12,
            profiles=("temporal_locator", "shared_entity_comparison"),
        )
        operations = [record["_taxonomy"].answer_operation for record in selected]
        # Pool is 2/3 identify, 1/3 compare. Sequential-id fill would keep almost
        # only identify after the two profile seeds.
        self.assertGreaterEqual(operations.count("compare_change"), 3)
        self.assertGreaterEqual(operations.count("identify"), 6)


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


class KimiAnnotatorTests(unittest.TestCase):
    def test_requires_api_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "MOONSHOT_API_KEY"):
            KimiOracleAnnotator(api_key=None)

    def test_response_format_matches_shared_schema(self) -> None:
        annotator = KimiOracleAnnotator(api_key="test-key")
        for mode in ("query_conditioned", "query_agnostic"):
            response_format = annotator._response_format(mode)
            self.assertEqual(response_format["type"], "json_schema")
            self.assertEqual(
                response_format["json_schema"]["schema"], gemini_response_schema(mode)
            )

    def test_query_conditioned_requires_question(self) -> None:
        annotator = KimiOracleAnnotator(api_key="test-key")
        with self.assertRaisesRegex(ValueError, "requires a question"):
            annotator.annotate(
                sample_id="s1",
                video_path="/tmp/does-not-need-to-exist.mp4",
                question="   ",
                annotation_mode="query_conditioned",
            )

    def test_error_carries_last_payload(self) -> None:
        error = KimiAnnotationError(
            "boom", raw_response="{bad json", parsed_events=[{"event_id": "e1"}]
        )
        self.assertEqual(error.raw_response, "{bad json")
        self.assertEqual(error.parsed_events, [{"event_id": "e1"}])


class ProviderCliTests(unittest.TestCase):
    def test_provider_defaults_to_gemini(self) -> None:
        args = build_parser().parse_args(["--output-dir", "/tmp/out"])
        self.assertEqual(args.provider, "gemini")
        self.assertIsNone(args.model)

    def test_kimi_provider_selectable(self) -> None:
        args = build_parser().parse_args(
            ["--output-dir", "/tmp/out", "--provider", "kimi", "--model", "kimi-k2.6"]
        )
        self.assertEqual(args.provider, "kimi")
        self.assertEqual(args.model, "kimi-k2.6")


class VerificationTests(unittest.TestCase):
    OPTIONS = (
        "A. A stone wall on the right side of the paved area near the parking lot. "
        "B. A brown trash can on the left side of the path by the curved road. "
        "C. A concrete bench on the left side of the path. "
        "D. A wooden post standing at the edge of the paved walkway."
    )

    def test_parses_options_containing_periods(self) -> None:
        options = parse_mcq_options(self.OPTIONS)
        self.assertEqual(sorted(options), ["A", "B", "C", "D"])
        self.assertEqual(options["C"], "A concrete bench on the left side of the path.")
        self.assertTrue(options["A"].startswith("A stone wall"))

    def test_shuffle_is_deterministic_and_reversible(self) -> None:
        options = parse_mcq_options(self.OPTIONS)
        presented, mapping = shuffled_options(options, seed="510")
        self.assertEqual(presented, shuffled_options(options, seed="510")[0])
        self.assertEqual(sorted(mapping.values()), ["A", "B", "C", "D"])
        for letter, text in presented.items():
            self.assertEqual(text, options[mapping[letter]])

    def test_coverage_ratio_merges_overlapping_events(self) -> None:
        first = _event(event_id="e1", start_sec=0.0, end_sec=10.0)
        overlapping = _event(event_id="e2", start_sec=5.0, end_sec=12.0)
        disjoint = _event(event_id="e3", start_sec=20.0, end_sec=23.0)
        annotation = replace(_annotation(), events=(first, overlapping, disjoint))
        # Union is [0,12) and [20,23) = 15s of a 30s video, not 10+7+3=20s.
        self.assertAlmostEqual(coverage_ratio(annotation), 0.5, places=5)

    def test_coverage_ratio_drops_umbrella_parents(self) -> None:
        parent = _event(
            event_id="coarse",
            start_sec=0.0,
            end_sec=30.0,
            granularity="coarse",
            query_roles=("evidence",),
        )
        child = _event(
            event_id="fine",
            start_sec=10.0,
            end_sec=12.0,
            granularity="fine",
            query_roles=("answer_bearing",),
        )
        annotation = replace(_annotation(), events=(parent, child))
        # Only the leaf [10,12) counts: 2s / 30s, not the 30s wrapper.
        self.assertAlmostEqual(coverage_ratio(annotation), 2 / 30, places=5)

    def test_verdict_requires_checkable_citations(self) -> None:
        def verdict(**overrides):
            values = {
                "correct": True,
                "sufficiency": "sufficient",
                "cited": ["e1"],
                "unknown": [],
                "coverage": 0.05,
                "cited_answer_bearing": True,
            }
            values.update(overrides)
            return decide_verdict(**values)

        self.assertEqual(verdict(), "pass")
        self.assertEqual(verdict(sufficiency="partial"), "uncertain")
        self.assertEqual(verdict(cited=["e9"], unknown=["e9"]), "uncertain")
        self.assertEqual(verdict(correct=False), "fail")

    def test_high_coverage_pass_is_uninformative(self) -> None:
        # A crop covering most of the video approximates the full-video ceiling.
        self.assertEqual(
            decide_verdict(
                correct=True,
                sufficiency="sufficient",
                cited=["e1"],
                unknown=[],
                coverage=0.92,
                cited_answer_bearing=True,
            ),
            "uninformative",
        )
        # Coverage cannot rescue a wrong answer.
        self.assertEqual(
            decide_verdict(
                correct=False,
                sufficiency="sufficient",
                cited=["e1"],
                unknown=[],
                coverage=0.92,
                cited_answer_bearing=True,
            ),
            "fail",
        )

    def test_answer_from_anchor_only_citation_is_uncertain(self) -> None:
        # Sample 73 pattern: answered confidently while citing only the anchor.
        self.assertEqual(
            decide_verdict(
                correct=True,
                sufficiency="sufficient",
                cited=["event_1_large_star"],
                unknown=[],
                coverage=0.2,
                cited_answer_bearing=False,
            ),
            "uncertain",
        )

    def test_response_schema_restricts_to_presented_letters(self) -> None:
        schema = verification_response_schema(["A", "B", "C"])
        self.assertEqual(schema["properties"]["selected_option"]["enum"], ["A", "B", "C"])
        self.assertIn("supporting_event_ids", schema["required"])

    def test_prompt_withholds_ground_truth(self) -> None:
        options = parse_mcq_options(self.OPTIONS)
        prompt = build_verification_prompt("Which structure?", options, n_clips=3)
        self.assertIn("only the clips shown", prompt)
        self.assertIn("sufficiency", prompt)
        self.assertNotIn("ground truth", prompt.lower())


if __name__ == "__main__":
    unittest.main()
