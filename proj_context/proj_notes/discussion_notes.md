# Discussion Notes — Open Questions & Framing

Scratch space for ideas we're still deciding on. Not settled design. Fuller context in `proj_description.md` and `bites.md`.

## "Meaning of memory nodes" — reframe (RQ2/RQ3)

Original RQ2 phrasing ("dynamically change the **meaning** of memory nodes based on what the user is looking for") is ambiguous. Cleaner to split into **two distinct levers**:

- **Node-level (facet salience):** given a query, emphasize *which facets* of an event node are relevant (e.g. appearance vs. action vs. location) — **without** changing the node's underlying meaning/embedding. This is re-weighting/selection over an already-encoded node.
- **Edge-level (query-adaptive traversal):** given a query, modulate **edge weights** so retrieval traverses the graph along query-relevant relations. This is where cross-event structure gets used.

Keeping these separate avoids conflating "re-encode the node" with "re-route the traversal."

## What we're doing *now* vs. later

- **Now (primary design):** **query-adaptive graph traversal / node retrieval** — build the event graph, modulate edges by the query, retrieve a coherent connected set of events. Node meanings/embeddings stay fixed; the query acts on *structure and selection*, not on the embedding itself.
- **Later / keep in mind (not current focus):** actually **transforming the meaning of a node embedding** per query (RQ3's "general manifold → task-specific via learnable transform"). Interesting, but out of scope for the current design. Parking it here so we don't lose it.

## Why this matters

- Query-adaptive *traversal* is a lighter, more testable claim: does modeling + traversing inter-event relations beat similarity-only retrieval? (RQ1)
- Re-encoding node meaning is a heavier, separate research bet — worth revisiting only if traversal alone proves insufficient.
