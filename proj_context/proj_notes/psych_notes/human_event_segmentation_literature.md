# Human Event Segmentation — Psychology Literature Notes

Last updated: 2026-07-17

## Scope

These notes summarize psychology and cognitive-science literature relevant to
event segmentation, event memory, and hierarchical representations. They are
kept separate from the Gemini annotation design in
`[../../proj_design/gemini_events/gemini_event_annotation_design.md](../../proj_design/gemini_events/gemini_event_annotation_design.md)`.

The motivating question for MorphGraph is whether boundaries produced from
temporal self-similarity (TSM) over video embeddings correspond to useful
human-like event units. Longer-term, the goal is a Video-LLM that builds its
own memory by producing hierarchical event keys and boundaries while retaining
detailed frame features as values and associating events through temporal and
semantic relations.

## Main conclusions

1. There is usually no unique correct segmentation. People represent events at
  multiple temporal grains, and their task, knowledge, and goals influence
   where they place boundaries.
2. Fine and coarse events are often hierarchically related, but hierarchy
  should not be imposed through fixed duration rules or a rigid
   object-versus-action split.
3. Event categories are better represented as overlapping dimensions than as a
  mutually exclusive taxonomy.
4. A boundary is best understood as an event-model update. Changes in actions,
  objects, actors, locations, goals, causal structure, and perceptual features
   can all contribute to such an update.
5. Query grounding is cognitively plausible as a top-down attention signal,
  but query-conditioned events are not a universal segmentation.

## Event Segmentation Theory

Event Segmentation Theory (EST) proposes that observers maintain an active
event model that predicts what will happen next. When prediction error rises,
the model is updated; this update is perceived as an event boundary. Event
models combine bottom-up perceptual information with top-down knowledge about
actors, objects, goals, and plans.

This framing is relevant to TSM because TSM detects local changes in
embedding-space self-similarity. It may therefore capture one source of
prediction failure, but it does not necessarily represent conceptual changes
in goals, causality, or persistent object identity. Boundary evaluation should
measure which kinds of human-like transitions TSM captures rather than only
asking whether all boundaries coincide.

Key references:

- Zacks, Speer, Swallow, Braver, and Reynolds (2007), [Event Perception: A
Mind-Brain Perspective](https://doi.org/10.1037/0033-2909.133.2.273).
Canonical EST account of predictive event models, transient prediction
error, and boundary-triggered updating.
- Reynolds, Zacks, and Braver (2007), [A Computational Model of Event
Segmentation From Perceptual
Prediction](https://doi.org/10.1080/15326900701399913). Formal predictive
model of segmentation.
- Zacks, Speer, and Reynolds (2009), [Segmentation in Reading and Film
Comprehension](https://doi.org/10.1037/a0015305). Studies how changes across
situation dimensions accumulate at event boundaries.

## Hierarchical and multiscale event structure

People can segment the same activity into fine and coarse units. Fine units
often cluster into larger coarse units, particularly for goal-directed
activity. For example:

- coarse: prepare dinner
- intermediate: prepare the vegetables
- fine: retrieve a carrot, chop it, and add it to the pan

Coarse and fine boundaries often occur near one another, but need not be
identical. Coarse boundaries may occur slightly after nearby fine boundaries
because evidence for a larger event transition takes time to accumulate.

Descriptions also differ by grain. Fine descriptions tend to use more precise
action verbs, whereas coarse descriptions can emphasize objects, goals, or
larger activities. This is a statistical tendency, not a rule that objects
must define coarse events and actions must define fine events.

Key references:

- Kurby and Zacks (2008), [Segmentation in the Perception and Memory of
Events](https://doi.org/10.1016/j.tics.2007.11.004). Concise review of
hierarchical segmentation and its consequences for memory.
- Zacks, Tversky, and Iyer (2001), [Perceiving, Remembering, and Communicating
Structure in Events](https://doi.org/10.1037/0096-3445.130.1.29). Evidence
for goal-directed partonomic hierarchies and action-object organization.
- Zacks and Tversky (2001), [Event Structure in Perception and
Conception](https://doi.org/10.1037/0033-2909.127.1.3). Foundational review of
event partonomies and representations.
- Hard, Lozano, and Tversky (2006), [Hierarchical Encoding of Behavior:
Translating Perception Into
Action](https://doi.org/10.1037/0096-3445.135.4.588). Goal-subgoal encoding
and observational learning.
- Hard, Recchia, and Tversky (2011), [The Shape of
Action](https://doi.org/10.1037/a0024310). Relations among physical changes,
attention, and perceived breakpoints.

## Situation dimensions and event categories

Situation-model research suggests that event representations track changes in
time, space, participants, objects, causality, and intentionality. These are
better understood as overlapping dimensions than as mutually exclusive event
classes.

Candidate dimensions:

- action, motion pattern, or manipulation
- object appearance, disappearance, persistence, or state change
- agent, participant, or interaction change
- spatial movement or changing relations among entities
- location or scene transition
- goal, intention, or subgoal change
- causal transition, enabling condition, interruption, success, or failure
- temporal discontinuity, repetition, or resumption

The semantic content of an event should be distinguished from evidence that a
boundary occurred. Boundary evidence may include:

- motion or low-level visual-feature change
- object or contact change
- actor or interaction change
- location or scene change
- goal or subgoal change
- causal or state change
- edit or camera discontinuity

An event can have several semantic dimensions and several boundary cues.
Parent-child links can express temporal containment, while separate typed
links can express semantic composition, shared entities, causality,
recurrence, and other cross-event associations.

Relevant reference:

- Zwaan and Radvansky (1998), [Situation Models in Language Comprehension and
Memory](https://doi.org/10.1037/0033-2909.123.2.162). Motivates tracking time,
space, protagonists/entities, causality, and intentionality.

## Task and query dependence

Top-down knowledge and task goals affect human segmentation. An observer
looking for a particular object or trying to remember the steps of an activity
may attend to different transitions than an observer watching without a task.
This makes query grounding cognitively defensible, but it changes the target:
a query-conditioned segmentation should not be treated as the only universal
event structure.

A useful distinction for future experiments is:

- task-independent event structure: the general event backbone perceived in
the video
- task-conditioned salience: which events, entities, and relations matter for
the current query

This distinction aligns with a memory system in which stable event nodes can
have query-dependent facet salience and traversal without changing their
underlying identity.

## Boundary agreement and evaluation

Exact timestamp equality is not a useful primary metric. Human boundaries can
be gradual or ambiguous, and annotators may agree on the existence of a
transition while differing on its precise timestamp.

Relevant measurement ideas:

- one-to-one boundary precision, recall, and F1 within fixed tolerances
- boundary-similarity measures that give graded credit for nearby boundaries
- human-consensus density rather than premature binarization
- inter-annotator agreement reported separately by event grain
- event-count ratios to expose over- or under-segmentation
- segment overlap or covering scores
- coarse-fine enclosure and alignment
- chance-robust random and uniform-spacing baselines

Generic Event Boundary Detection (GEBD) uses relative distance: absolute
timestamp error divided by video duration, followed by precision, recall, and
F1 at a chosen threshold. This is useful for comparison with prior work but
can be too permissive for very long videos.

Relevant references:

- Shou et al. (2021), [Generic Event Boundary Detection: A Benchmark for Event
Segmentation](https://arxiv.org/abs/2101.10511).
- Ding, Sener, and Yao (2023), [Temporal Action Segmentation: An Analysis of
Modern Techniques](https://arxiv.org/abs/2210.10352). Reviews segmental F1,
temporal IoU, and edit metrics.
- Fournier (2013), [Evaluating Text Segmentation Using Boundary Edit
Distance](https://doi.org/10.3115/v1/P13-1167). Boundary similarity gives
graded credit for nearby boundaries.
- Sasmita and Swallow (2023), [Measuring Event Segmentation: An Investigation
Into the Stability of Event Boundary Agreement
Measures](https://doi.org/10.3758/s13428-022-01832-5). Discusses consensus
and agreement measurements.
- [Margin MCC: Chance-Robust Metric for Video Boundary
Detection](https://doi.org/10.1145/3581783.3613760). Shows that tolerance F1
can reward dense or random predictions and motivates chance-robust
evaluation.

## Related video annotation precedents

These are not psychology studies, but they provide useful operationalizations
of hierarchical or compositional event structure:

- Song et al. (2023), [Ego4D
Goal-Step](https://proceedings.neurips.cc/paper_files/paper/2023/hash/7a65606fa1a6849450550325832036e5-Abstract-Datasets_and_Benchmarks.html).
Uses nested goals, steps, and substeps, with a taxonomy refined from data
rather than completely prescribed in advance.
- Damen et al. (2018), [EPIC-KITCHENS](https://doi.org/10.1007/978-3-030-01225-0_43).
Uses compositional verb-noun action labels for egocentric activity.
- Ji, Krishna, Fei-Fei, and Niebles (2020), [Action
Genome](https://doi.org/10.1109/CVPR42600.2020.01025). Represents actions
through changing actor-object relations in spatiotemporal scene graphs.
- Gu et al. (2018), [AVA](https://doi.org/10.1109/CVPR.2018.00633). Annotates
simultaneous, multi-label atomic actions.
- Sigurdsson et al. (2016), [Hollywood in Homes /
Charades](https://arxiv.org/abs/1604.01753). Includes overlapping temporal
actions composed from verbs, objects, and relations.

Together, these datasets support overlapping event tracks and compositional
representations rather than one exclusive event class per interval.

## Implications for MorphGraph

- Evaluate TSM against fine and coarse boundaries separately.
- Represent event categories as overlapping channels.
- Keep event semantics separate from the cue that triggered a boundary.
- Allow temporal containment and semantic association to coexist.
- Test query-independent structure separately from query-conditioned utility.
- Do not tune TSM on the small pseudo-annotation pilot.
- Retain frame-level values even if learned hierarchical event keys become the
primary memory index.

## Open research questions

- Which conceptual boundaries are visible in current Qwen3-VL pooled features,
and which require prediction, language, or explicit entity tracking?
- Should hierarchical event keys be learned as separate latent tokens at each
grain, or as category-specific views over shared frame values?
- Can boundary types be learned as overlapping channels rather than a single
multiclass output?
- Which event associations most improve memory retrieval: temporal adjacency,
shared entities, recurrence, causality, goals, or spatial continuity?
- Does query grounding improve annotation completeness while preserving a
stable query-independent event backbone?
- How much apparent TSM agreement remains after controlling for random or
uniformly spaced boundaries?

