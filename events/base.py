"""Model-agnostic interface for event segmentation.

An event segmenter consumes a stream of per-unit embeddings (e.g.
`VisualFeatures.pooled`, shape `(n_units, dim)`) and returns event boundaries +
half-open unit spans. Kept separate from feature extraction so any boundary
method (TSM novelty, adjacency, Otsu, ...) can plug in.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass
class EventSegmentation:
    # Unit indices that START a new event (excludes 0).
    boundaries: list[int]
    # Half-open `(start_unit, end_unit)` spans covering all units.
    events: list[tuple[int, int]]
    # Per-unit novelty/score curve (for viz/debug); 0 at invalid edges.
    novelty: np.ndarray
    n_units: int

    @property
    def n_events(self) -> int:
        return len(self.events)


class EventSegmenter(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def segment(self, pooled: Any) -> EventSegmentation:
        """Segment per-unit embeddings `(n_units, dim)` into events."""


def events_from_boundaries(boundaries: list[int], n_units: int) -> list[tuple[int, int]]:
    """Turn sorted boundary unit-indices into contiguous half-open spans."""
    cuts = [0, *[int(b) for b in boundaries if 0 < b < n_units], n_units]
    cuts = sorted(set(cuts))
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]


# --- registry: any segmentation method plugs in under a string key -----------

_SEGMENTERS: dict[str, Callable[..., EventSegmenter]] = {}


def register_segmenter(name: str, factory: Callable[..., EventSegmenter]) -> None:
    _SEGMENTERS[name] = factory


def get_segmenter(name: str, **kwargs) -> EventSegmenter:
    if name not in _SEGMENTERS:
        raise KeyError(f"Unknown segmenter {name!r}. Registered: {sorted(_SEGMENTERS)}")
    return _SEGMENTERS[name](**kwargs)
