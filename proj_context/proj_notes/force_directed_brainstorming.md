# Force-Directed Graph Memory — Brainstorming

## Core idea

Treat video memory as a dynamic graph rather than a fixed write-time index.
Event and entity nodes are connected by typed, signed spring edges:

- semantic similarity and temporal continuity create attraction;
- contradiction, dissimilarity, or competition create repulsion;
- edge strengths change as new evidence arrives.

After each update, the graph relaxes toward an equilibrium. The intended result
is a self-organizing memory topology: mutually supporting events form stable
regions, while conflicting or unrelated events separate.

## Proposed model

Each node has:

- an event/entity representation and metadata;
- a retrieval key embedding;
- a graph position used only for memory organization/traversal;
- optional confidence, recency, and retrieval-outcome statistics.

Each edge has:

- a relation type: `temporal`, `semantic`, `entity`, `conflict`, or
  `co_retrieval`;
- a signed weight: attraction (`> 0`) or repulsion (`< 0`);
- provenance: which update or feedback signal created it;
- decay/confirmation state.

The position update should use a bounded incremental layout rather than
re-running a full force-directed solver over every historical event:

```text
new event
  → retrieve local candidate neighborhood
  → add typed attraction/repulsion edges
  → relax only the affected k-hop subgraph for a fixed number of iterations
  → persist edges, positions, and update provenance
```

## Retrieval hypothesis

Instead of returning only vector top-k seeds:

1. obtain vector seeds from the query;
2. initiate a query-conditioned traversal over the local equilibrium graph;
3. accumulate nodes supported by attractive paths and penalize paths crossing
   conflict/repulsive edges;
4. stop once score mass or frame/token budget is exhausted;
5. return event frames plus path/edge provenance for retrieval evaluation.

This should be compared against the current vector-seed plus one-hop expansion
baseline, not assumed to be better.

## Potential signals

| Signal | Proposed effect | Open concern |
| --- | --- | --- |
| high semantic similarity | attraction | may duplicate vector k-NN edges |
| temporal adjacency | weak attraction | should not collapse an entire video into one cluster |
| shared tracked entity | attraction | entity identity must be reliable |
| incompatible scene/object state | repulsion | need a robust conflict detector |
| retrieved together and QA succeeds | reinforce traversed path | avoid self-reinforcing retrieval bias |
| retrieved but QA fails / evidence absent | downweight or inspect path | answer failure may be a generator failure, not retrieval failure |
| recency or task scope | query-time bias, not necessarily a permanent edge | avoid rewriting memory for one question |

## Why it may be useful

- Encodes long-range recurrence beyond adjacent event links.
- Makes retrieval paths inspectable: why a node was included can be attributed to
  seeds and typed edges.
- Provides a natural place to use retrieval-grounded feedback after we can
  measure event-frame recall/precision.
- Connects to FluxMem's idea of feedback-driven link expansion and pruning
  without adopting its agent/procedural-memory stack.

## Potential advantages over a static graph

- **Emergent organization.** Closeness is the result of attraction/repulsion
  equilibrium, rather than a topology imposed once at write time. This may let
  event clusters emerge without hard-coding all adjacency decisions.
- **Separate salience from similarity.** A node's Coulomb-style charge can
  encode salience or associability independently of distance/semantic
  compatibility. Standard attention or k-NN generally entangles those signals
  into one score.
- **Memory plasticity and consolidation.** Recent edges can remain elastic
  while older, confirmed relations stiffen. This creates a concrete mechanism
  for temporal plasticity instead of treating all historical edges identically.
- **Principled cluster-granularity analysis.** Specific force-directed energy
  models are related to graph clustering/modularity objectives (for example,
  Noack, 2009). The chosen repulsion model therefore encodes an implicit
  preference for coarse versus fine event communities, which can be evaluated
  rather than treated as an arbitrary threshold.
- **Beyond linear graph diffusion.** Bounded nonlinear repulsion and nonzero
  rest lengths would not reduce to ordinary Personalized PageRank or Laplacian
  diffusion. That distinction matters if a force model is proposed as more than
  a visualization of the current graph.
- **Biological narrative.** The idea has a plausible analogy to hippocampal
  pattern separation, attractor dynamics, and repulsion between confusable
  memories. This is motivation, not evidence that the mechanism improves video
  QA.

## Design tensions and risks

- **Layout is not edge generation.** Keep the energy/layout mechanism separate
  from the model that proposes typed edges and their initial weights. Otherwise
  it becomes unclear whether a result is caused by learned relations or by the
  relaxation procedure.
- **Differentiability.** A true iterative equilibrium solve is not a standard
  single differentiable forward pass. Learning through it requires implicit
  differentiation or a surrogate that learns force parameters without
  backpropagating through the full trajectory.
- **Resolution limits.** Some degree-weighted/modularity-like repulsion
  formulations fail to separate small communities; node-count or
  constant-Potts-model-style alternatives change that bias. The force law must
  be selected against target event-boundary granularity.
- **Unclear charge supervision.** Annotation density, novelty, retrieval
  frequency, and QA utility are only proxies for salience. It is unresolved
  whether charge should be static, time-decayed, or updated from validated
  retrieval outcomes.
- **Streaming compute cost.** Full equilibrium after each update is likely too
  expensive for long video. Any viable version needs bounded local relaxation,
  warm starts, and a measured update budget.
- **Debuggability.** Emergent structure can be less directly attributable than
  an explicit learned adjacency matrix. Every update should log proposed edges,
  force parameters, affected nodes, pre/post positions, and retrieval paths.

## Risks and non-goals

- A physical layout is not automatically a better similarity index. High
  dimensional embeddings and typed graph traversal may be sufficient without
  force coordinates.
- The Noack layout/clustering connection is force-model-specific; it should not
  be claimed as an equivalence for arbitrary signed springs or nonlinear
  repulsion.
- Signed force systems can be unstable, sensitive to scale, and expensive to
  update. Use bounded local relaxation and retain the original vector index as
  a fallback.
- Repulsion must not delete evidence. It should influence traversal/ranking,
  while source events remain recoverable.
- Do not introduce answer-based graph rewrites until retrieval and generation
  errors can be separated with retrieval labels or human inspection.
- This is not a replacement for the current temporal/semantic EventGraph
  baseline; it is an evaluable experimental branch.

## First minimal experiment

1. Keep the current event nodes and vector seeds.
2. Add only positive temporal and semantic springs; no repulsion yet.
3. Locally relax positions when inserting a video.
4. Compare equilibrium-neighborhood expansion with current one-hop expansion
   using saved retrieved event-frame traces.
5. Add one carefully defined repulsive signal only if the positive-edge
   baseline shows a measurable failure mode (for example, recurring but
   semantically incompatible events crowding the same retrieval set).

## Questions to resolve

- Are positions a retrieval feature, a visualization/diagnostic tool, or both?
- What qualifies as a conflict in video memory: visual contradiction, temporal
  exclusivity, entity-state change, or question-conditioned competition?
- Should a query alter persistent edges, or only use transient query-time
  forces?
- Which retrieval metric determines whether an edge update was beneficial?
- Can entity nodes be introduced before tracking/identity quality is validated?
