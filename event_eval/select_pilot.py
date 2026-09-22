"""Select a deterministic pilot using the query-structure dashboard taxonomy."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .annotate import DEFAULT_QUESTIONS
from .query_taxonomy import QueryTaxonomy, classify_question

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DASHBOARD = (
    _PACKAGE_ROOT
    / "proj_context"
    / "benchmark_notes"
    / "egolongqa_dashboard_data.json"
)
DEFAULT_PROFILES = (
    "temporal_locator",
    "shared_entity_identification",
    "shared_entity_comparison",
    "shared_location",
    "shared_activity_goal",
    "causal_dependency",
    "multi_fact_combine",
    "explicit_order",
)


def _sample_ids_by_video(question_path: Path) -> dict[str, str]:
    result = {}
    with question_path.open() as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            result[str(row["video_path"])] = str(row.get("qid", row.get("id", index)))
    return result


def select_records(
    records: list[dict[str, Any]],
    *,
    count: int,
    profiles: tuple[str, ...] = DEFAULT_PROFILES,
) -> list[dict[str, Any]]:
    if count < 1:
        raise ValueError("count must be positive")
    classified = [
        (record, classify_question(str(record["question"]))) for record in records
    ]
    selected: list[dict[str, Any]] = []
    selected_videos: set[str] = set()
    selected_scopes: set[str] = set()
    selected_links: set[str] = set()
    selected_operations: set[str] = set()
    selected_categories: set[str] = set()

    def add(record: dict[str, Any], taxonomy: QueryTaxonomy) -> None:
        video = str(record["video"])
        if video not in selected_videos and len(selected) < count:
            selected.append({**record, "_taxonomy": taxonomy})
            selected_videos.add(video)
            selected_scopes.add(taxonomy.evidence_scope)
            selected_links.update(taxonomy.link_bases)
            selected_operations.add(taxonomy.answer_operation)
            selected_categories.add(str(record.get("category", "")))

    def matches(profile: str, taxonomy: QueryTaxonomy) -> bool:
        return {
            "temporal_locator": (
                taxonomy.evidence_scope == "single_target"
                and "temporal_constraint" in taxonomy.link_bases
                and taxonomy.answer_operation == "identify"
            ),
            "shared_entity_identification": (
                "shared_entity" in taxonomy.link_bases
                and taxonomy.answer_operation == "identify"
            ),
            "shared_entity_comparison": (
                "shared_entity" in taxonomy.link_bases
                and taxonomy.answer_operation == "compare_change"
            ),
            "shared_location": "shared_location" in taxonomy.link_bases,
            "shared_activity_goal": "shared_activity_goal" in taxonomy.link_bases,
            "causal_dependency": "causal_dependency" in taxonomy.link_bases,
            "multi_fact_combine": (
                taxonomy.evidence_scope == "multi_event_integration"
                and taxonomy.answer_operation == "combine"
            ),
            "explicit_order": taxonomy.answer_operation == "order",
        }.get(profile, False)

    def diversity_score(record: dict[str, Any], taxonomy: QueryTaxonomy) -> tuple[int, int]:
        score = 3 * int(taxonomy.evidence_scope not in selected_scopes)
        score += 2 * len(set(taxonomy.link_bases) - selected_links)
        score += 2 * int(taxonomy.answer_operation not in selected_operations)
        score += int(str(record.get("category", "")) not in selected_categories)
        return score, -int(record.get("id", 0))

    for profile in profiles:
        candidates = [
            (record, taxonomy)
            for record, taxonomy in classified
            if str(record["video"]) not in selected_videos
            and matches(profile, taxonomy)
        ]
        if candidates and len(selected) < count:
            record, taxonomy = max(
                candidates, key=lambda pair: diversity_score(*pair)
            )
            add(record, taxonomy)

    def _stratum(taxonomy: QueryTaxonomy) -> tuple[str, str]:
        return (taxonomy.evidence_scope, taxonomy.answer_operation)

    def _largest_remainder_targets(
        counts: Counter[tuple[str, str]], slots: int
    ) -> dict[tuple[str, str], int]:
        total = sum(counts.values())
        if slots <= 0 or total <= 0:
            return {}
        raw = {key: slots * counts[key] / total for key in counts}
        floors = {key: int(raw[key]) for key in counts}
        assigned = sum(floors.values())
        remainders = sorted(
            counts,
            key=lambda key: (raw[key] - floors[key], counts[key], key),
            reverse=True,
        )
        targets = dict(floors)
        for key in remainders:
            if assigned >= slots:
                break
            targets[key] += 1
            assigned += 1
        return targets

    # Fill remaining slots so (evidence_scope, answer_operation) matches the
    # leftover pool, instead of walking lowest source IDs once every taxonomy
    # bucket has been seen once.
    if len(selected) < count:
        pool = [
            pair
            for pair in classified
            if str(pair[0]["video"]) not in selected_videos
        ]
        by_stratum: dict[tuple[str, str], list[tuple[dict[str, Any], QueryTaxonomy]]] = (
            defaultdict(list)
        )
        for record, taxonomy in pool:
            by_stratum[_stratum(taxonomy)].append((record, taxonomy))
        remaining_targets = _largest_remainder_targets(
            Counter({key: len(rows) for key, rows in by_stratum.items()}),
            count - len(selected),
        )
        while len(selected) < count and any(remaining_targets.values()):
            progressed = False
            for key in sorted(remaining_targets):
                if remaining_targets[key] <= 0 or len(selected) >= count:
                    continue
                candidates = [
                    pair
                    for pair in by_stratum[key]
                    if str(pair[0]["video"]) not in selected_videos
                ]
                if not candidates:
                    remaining_targets[key] = 0
                    continue
                record, taxonomy = max(
                    candidates, key=lambda pair: diversity_score(*pair)
                )
                add(record, taxonomy)
                remaining_targets[key] -= 1
                progressed = True
            if not progressed:
                break
        while len(selected) < count:
            leftover = [
                pair
                for pair in classified
                if str(pair[0]["video"]) not in selected_videos
            ]
            if not leftover:
                break
            record, taxonomy = max(leftover, key=lambda pair: diversity_score(*pair))
            add(record, taxonomy)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard-json", type=Path, default=DEFAULT_DASHBOARD)
    parser.add_argument("--question-path", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()

    dashboard = json.loads(args.dashboard_json.read_text())
    selected = select_records(dashboard["records"], count=args.count)
    ids_by_video = _sample_ids_by_video(args.question_path)
    missing = [str(record["video"]) for record in selected if str(record["video"]) not in ids_by_video]
    if missing:
        raise ValueError(f"Selected dashboard videos missing from question JSONL: {missing}")
    sample_ids = [ids_by_video[str(record["video"])] for record in selected]
    manifest = []
    for sample_id, record in zip(sample_ids, selected):
        taxonomy = record["_taxonomy"]
        manifest.append(
            {
                "sample_id": sample_id,
                "video": record["video"],
                "question": record["question"],
                "evidence_scope": taxonomy.evidence_scope,
                "link_bases": list(taxonomy.link_bases),
                "answer_operation": taxonomy.answer_operation,
            }
        )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, indent=2) + "\n")
    for sample_id, record in zip(sample_ids, selected):
        question = record["question"]
        taxonomy = record["_taxonomy"]
        print(
            f"{sample_id}: {record['video']} "
            f"{question} "
            f"scope={taxonomy.evidence_scope} links={list(taxonomy.link_bases)} "
            f"operation={taxonomy.answer_operation}"
        )
    print(f"Saved {len(sample_ids)} pilot records to {args.output_json}")


if __name__ == "__main__":
    main()
