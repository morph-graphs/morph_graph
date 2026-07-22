"""Evaluation of query-agnostic TSM segments against query-conditioned oracles.

Bootstrap the repository parent so ``python -m event_eval.<command>`` works
when invoked from the ``morph_graph/`` directory, matching the debug scripts.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPOSITORY_ROOT = str(Path(__file__).resolve().parents[2])
if _REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, _REPOSITORY_ROOT)

from .metrics import compare_annotation, compare_event, query_event_coverage
from .schema import (
    EVENT_CATEGORIES,
    GRANULARITIES,
    QUERY_ROLES,
    OracleAnnotation,
    OracleEvent,
)
from .query_taxonomy import QueryTaxonomy, classify_question

__all__ = [
    "EVENT_CATEGORIES",
    "GRANULARITIES",
    "QUERY_ROLES",
    "OracleAnnotation",
    "OracleEvent",
    "QueryTaxonomy",
    "classify_question",
    "compare_annotation",
    "compare_event",
    "query_event_coverage",
]
