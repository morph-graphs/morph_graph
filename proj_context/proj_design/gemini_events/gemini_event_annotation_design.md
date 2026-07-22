# Gemini Event Annotation — Evaluation Design

Last updated: 2026-07-18

## Goal

Use Gemini 3.1 Pro to produce a query-conditioned upper-bound event oracle for
evaluating whether query-agnostic TSM segments are useful memory primitives.

This is **evaluation supervision**, not the target architecture:

- **Eval:** Gemini sees the video and question; event selection and boundaries
  are optimized for answering that question.
- **Training/deployment:** memory events and base relationships remain
  query-agnostic. Query-time morphing only reweights salient facets and edges.

The oracle is intentionally minimal, not an exhaustive video segmentation. A
TSM event absent from the oracle may be valid but irrelevant to that question.

## What counts as an event

An event is a meaningful action, interaction, state change, goal/subgoal,
spatial or location transition, causal transition, or temporal transition.

Mere object visibility is not an event. An object becomes event-relevant only
through intentional interaction, state change, or meaningful context change.

Gemini may return fine events, coarse events, or both. For **query-agnostic**
segmentation (`eval_agnostic_v2`), fine/coarse are optional scale labels — not
duration tags — and hierarchy is rare. Most events should be roots; a coarse
event need not have children. Avoid micro-tiling (repeated near-duplicate
spans) and video-long umbrella parents.

## Output

Each event contains:

- `event_id`
- `start_sec`, `end_sec`
- `granularity`: `fine` or `coarse`
- grounded `description`
- multi-label `categories`
- `parent_event_ids`
- `query_roles`: `anchor`, `evidence`, and/or `answer_bearing`

Categories:

`action`, `object_or_state`, `agent_or_interaction`, `spatial_relation`,
`location_or_scene`, `goal_or_subgoal`, `causal_transition`,
`temporal_transition`.

Every returned event must have a query role:

- `anchor`: locates or constrains the target
- `evidence`: connects or disambiguates the answer
- `answer_bearing`: directly contains requested information

## Hierarchy

A parent-child link requires both:

1. **Temporal containment:** the full child interval lies inside the parent.
2. **Semantic composition:** the child meaningfully realizes or decomposes the
   parent's activity or objective.

Overlap or adjacency alone is insufficient. Code validates containment;
semantic composition is enforced through the prompt.

### Agnostic prompt failure modes (v1 → v2)

Pilot review found `eval_agnostic_v1` often:

- alternating near-duplicate micro-events (glue/strip, manual/pieces),
- labeling every span `coarse` (or fine) by length with no real hierarchy,
- one video-long coarse parent with dozens of children,
- parenting the *next* adjacent beat as a child.

`eval_agnostic_v2` explicitly forbids those patterns and asks for a compact
activity-level partition.

## Annotation process

1. Select a small pilot using evidence scope, link basis, and answer operation.
2. Upload one video and question to `gemini-3.1-pro-preview`.
3. Request schema-constrained JSON at temperature `0`, sampled at `1 FPS`.
4. Validate timestamps, IDs, categories, query roles, and hierarchy.
5. Preserve the parsed annotation, raw response, model/prompt version, and
   sampling configuration.
6. Freeze the protocol after the pilot, then scale to all 700 questions.

The runner is resumable and records explicit failures rather than silently
repairing invalid outputs.

## TSM evaluation

Primary segmentation metrics compare TSM to the **query-agnostic** Gemini
partition (per video). The query-conditioned oracle remains an upper-bound
for question-specific analysis, not the default TSM score reference.

Primary metrics (agnostic reference):

- **Boundary error:** reference start/end distance from covering TSM
  boundaries.
- **Over-segmentation:** extra TSM segments per reference event
  (`max(0, n_tsm ∩ ref − 1)`).
- **Under-segmentation:** extra reference events per TSM segment that hits
  any reference (`max(0, n_ref ∩ TSM − 1)`).
- **Contamination:** unrelated duration included by that minimal TSM cover.

Supplemental diagnostics include one-to-one boundary F1 at 2/5/10 seconds
(prefer 2s) and random/uniform baselines. Query-event coverage belongs to
retrieval evaluation; the complete TSM partition trivially covers the full
video.

Evaluate the current TSM defaults first without tuning on the pilot:
`kernel_width=6`, `peak_height=1.0`, normalized novelty, post-projection
Qwen3-VL features.

## Main caveats

- Gemini's 1 FPS sampling may miss rapid boundaries.
- Prompt wording affects granularity.
- Goals and causal links may be inferred rather than directly visible.
- Model aliases can change, so exact provenance must be stored.
- These are pseudo-reference annotations, not human ground truth.

## Links

- [Implementation and commands](../../../event_eval/README.md)
- [Psychology literature](../../proj_notes/psych_notes/human_event_segmentation_literature.md)
- [Gemini video understanding](https://ai.google.dev/gemini-api/docs/video-understanding)
- [Gemini structured outputs](https://ai.google.dev/gemini-api/docs/structured-output)
