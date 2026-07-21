"""Automatic question taxonomy shared by pilot selection and the dashboard design."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryTaxonomy:
    evidence_scope: str
    link_bases: tuple[str, ...]
    answer_operation: str


_MULTI_EVIDENCE = (
    re.compile(r"\b(different|difference|changed|change|compared)\b", re.I),
    re.compile(
        r"\b(first time|initially|earlier)\b.*\b(last time|second time|later|eventually)\b",
        re.I,
    ),
    re.compile(r"\b(same|again|twice)\b", re.I),
    re.compile(r"\b(that|which|it)\b.*\b(later|earlier|again|eventually)\b", re.I),
    re.compile(r"\b(how many|total|sum|number of years|duration)\b", re.I),
    re.compile(r"\bwhat\b.*\band\b.*\bwhat\b", re.I),
)

_LINK_RULES = (
    (
        "temporal_constraint",
        re.compile(
            r"\b(after|before|between|first|last|earlier|later|when|while|during|then)\b",
            re.I,
        ),
    ),
    (
        "shared_entity",
        re.compile(
            r"\b(same|again|twice|second time|first time.*last time|"
            r"(that|which|it)\b.*\b(later|earlier|again))\b",
            re.I,
        ),
    ),
    (
        "shared_location",
        re.compile(
            r"\b(same (place|location|area|room)|returned? to|back (at|to|in)|"
            r"location.*again)\b",
            re.I,
        ),
    ),
    (
        "shared_activity_goal",
        re.compile(
            r"\b(same (activity|task|project)|continued|resumed|finished|completed)\b",
            re.I,
        ),
    ),
    (
        "causal_dependency",
        re.compile(
            r"\b(why|caused|prompted|reason|because|led to|evidence indicates)\b",
            re.I,
        ),
    ),
)


def classify_question(question: str) -> QueryTaxonomy:
    """Classify explicit query structure without inspecting the video."""
    evidence_scope = (
        "multi_event_integration"
        if any(pattern.search(question) for pattern in _MULTI_EVIDENCE)
        else "single_target"
    )
    links = tuple(
        name for name, pattern in _LINK_RULES if pattern.search(question)
    ) or ("unspecified_link",)

    if re.search(
        r"\b(how many|total|sum|number of years|duration|how long)\b", question, re.I
    ):
        operation = "aggregate"
    elif re.search(
        r"\b(why|caused|prompted|reason|because|what evidence|indicates)\b",
        question,
        re.I,
    ):
        operation = "explain"
    elif re.search(
        r"\b(different|difference|changed|change|compared|"
        r"first time.*last time|initially.*later)\b",
        question,
        re.I,
    ):
        operation = "compare_change"
    elif re.search(
        r"\b(in what order|what (?:was|is) the order|what sequence)\b",
        question,
        re.I,
    ) or re.search(
        r"\bwhich\b.*\b(?:did i|was)\b.*\bfirst\b.*\b(?:or|before)\b",
        question,
        re.I,
    ):
        operation = "order"
    elif re.search(r"\bwhat\b.*\band\b.*\bwhat\b", question, re.I) or re.search(
        r"\bwhich\b.*\band\b.*\bwhere\b", question, re.I
    ):
        operation = "combine"
    else:
        operation = "identify"
    return QueryTaxonomy(evidence_scope, links, operation)
