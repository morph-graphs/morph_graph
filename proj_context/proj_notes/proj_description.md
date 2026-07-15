# Morph Graphs: Query-Adaptive Event-Graph Memory for Video Understanding

## Background

How prior work remembers a video can be read along **two orthogonal axes**: *how* a unit is stored, and *where* (in which representational space) the memory is constructed.

**Axis 1 — How memory is stored/accessed** (five families; see `bites.md`):

- **A — Retrieval over stored KV:** keep per-frame/clip KV (unbounded, offloaded), retrieve query-relevant blocks. *(e.g. ReKV)*
- **B — Fixed-size / adaptive token compression:** compress into a bounded (or adaptively thresholded) budget by pruning/merging redundancy. *(fixed-size KV: StreamMem, HERMES; adaptive token: FluxMem)*
- **C — Consolidated / recurrent memory banks:** merge history into learned memory or short→long-term banks. *(e.g. MovieChat, Flash-VStream)*
- **D — Symbolic text + object memory:** store captions / object records in language space, queried by tools. *(e.g. VideoAgent)*
- **E — Structured graph memory (ours):** events linked by relational edges, traversed under the query.

**Axis 2 — Where memory is constructed** (cuts across the families above):

| Memory space | Operates on | Examples |
| --- | --- | --- |
| **Encoder / visual-feature space** (pre-LLM) | visual tokens / feature maps *before* the LLM | Flash-VStream, FluxMem, MovieChat, LongVU |
| **LLM KV-cache space** | KV entries inside/after LLM attention | ReKV, StreamMem, HERMES |
| **Symbolic / language space** | captions, object records | VideoAgent |
| **Relational / graph space** (ours) | events + edges | Morph Graphs |

Two clarifications this axis makes explicit:

- **Flash-VStream** (Flash Memory: CSM context + DAM detail) and **FluxMem** (TAS/SDC token prune+merge) build/compress memory in the **encoder/visual-feature space *before* the LLM** — categorically different from KV-cache compression (StreamMem, HERMES), which operates *inside* the LLM. So FluxMem sits next to Flash-VStream on Axis 2, even though it is compression-based (family B) on Axis 1.
- The two axes are independent: e.g. MovieChat is a *consolidated bank* (family C) that also lives in *encoder space* (Axis 2).

## Problem

Existing video memory — across every family and representational space above — represents remembered units **independently** and accesses them by **query-to-unit similarity**.

**Gap:** none model **inter-event relations** as first-class structure that retrieval can traverse. This is orthogonal to *how* or *where* memory is compressed; even the graph-adjacent work (Vgent, VideoDetective) does not modulate relational structure by the query.

This caps **multi-hop QA**, where answering requires aggregating scattered evidence across the video. Similarity-only retrieval then:

- returns a scattered set of moments rather than a coherent sequence, and
- misses contextually related events that are relevant but not directly similar to the query.

## Motivation

Inspired by **human memory**, which:

- Learns to separate continuous experience into discrete **"events"**, characterized by:
  - **Intrinsic contexts**: goals, actions
  - **Extrinsic contexts**: location, spatial surroundings
- Is able to **link events together** based on these shared contexts.

## Key Research Questions

1. **(Primary)** Is graph-based memory better than raw similarity retrieval? Does modeling + traversing **inter-event relations** benefit understanding?
2. Given a query, can we **re-weight which facets of an event node are salient** (appearance vs. action vs. location) — selection over a fixed node, *not* changing its embedding?

*Parked (not current focus; see `discussion_notes.md`):* whether to **transform node meaning** per query via a learnable general→task-specific manifold. Kept in mind, out of scope for now — the current design acts on graph **structure/traversal**, not on node embeddings.

## Proposed Idea

- Build memory as a **structured event graph**, linking dynamic sets of frames (**events**) based on shared context.
  - "Events" is an overarching term here, implying objects, scenes, actions, etc.
- **Query-Adaptive Traversal and Retrieval**:
  - The constructed graph should be **dynamic**, such that **edge weights are modulated by the query**.

### Motivating Examples

- How did the man's appearance change from the beginning of the day to the end?
- When were the man and the dog together?
- What happened to the man after the fire alarm went off?
