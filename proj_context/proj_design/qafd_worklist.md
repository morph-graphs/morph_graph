# QAFD — Issue Worklist

Ordered checklist for auditing and fixing the QAFD retriever. Companion to
[qafd_retrieval.md](./qafd_retrieval.md) (the math) — this is the "what's
actually wrong and why" list.

**Status: nothing here has been empirically verified.** Every item comes from
reading `graphrag/vl_graphrag/qafd_retriever.py` and diffing it against the
upstream clone at `/fs/cfar-projects/mem_open_vocab/morph_graphs/QAFD-RAG`.
Items marked *(predicted)* are consequences derived from the code, not measured.

Line numbers are hints — function names are the stable reference.

---

## Part A — Orientation (read-only, no changes)

Do these first. They're what makes the rest legible.

### A1. Read the two implementations side by side

- [ ] Ours: `graphrag/vl_graphrag/qafd_retriever.py` (320 lines, whole file)
- [ ] Upstream: `QAFD-RAG/src/passage_entity/graph_adapter.py::IGraphQAFD` (lines 77–395)

They are a near line-for-line match (networkx/dicts vs igraph/numpy). Confirm
that for yourself — it's what lets you trust upstream as the reference for
every parameter question later.

**What to notice:** upstream's `_push` (line 304) and ours (line 228) have the
same three-phase structure, and both carry the same comment about decoupling
accumulation from routing.

### A2. Trace one query end to end

- [ ] `vl_ops.py::vl_query` → the `elif param.mode == "qafd"` branch (~line 207)
- [ ] Follow: `vdb.query(top_k)` → `seed_scores` → `NetworkXQAFDRetriever` → `retrieve()` → `rank_and_cap_node_ids`

**What to notice:** where the VDB scores go, and where they stop mattering.
This sets up D1.

### A3. Understand the two upstream entry points

- [ ] Old/simple: `QAFD-RAG/src/retrievers/flow_diffusion.py` — single source node, no `qa_*` knobs
- [ ] Production: `graph_adapter.py::IGraphQAFD` + `passage_entity/config.py` + `passage_entity/retriever.py`

Our port descends from the **production** one (it has all the `qa_*` knobs) but
inherited some values from the older one. Knowing there are two upstream
versions explains most of Part B.

---

## Part B — Parameter divergences from upstream

### B1. `alpha` is 10.0, upstream production runs 1.5 ⚠ **highest priority**

- [ ] Fix

- **Ours:** `vl_base.py:26` `qafd_alpha: float = 10.0`
- **Upstream:** `QAFD-RAG/src/passage_entity/config.py:53` `qafd_alpha: float = 1.5`
- **Why the mismatch:** `10.0` is the *class signature* default at
`graph_adapter.py:108`. Upstream never uses it — `retriever.py:336` passes
`config.qafd_alpha`. The port copied the signature and missed the override.
- **Why it matters:** sink capacity is pinned at $\sum_u T_u = 10$ and mass is
injected as $\alpha T s_u$, so the saturation ratio is exactly $\rho = \alpha$.
At $\alpha = 10$ on a 19-node graph you inject 100 units against ~0.53
capacity per node — total saturation.
- **Consequence (predicted):** once every node is saturated, $x_u$ grows by
$\eta r_u / W_u$ everywhere, i.e. **by degree**. Seed signal washes out and
ranking degenerates toward query-independent centrality.
- **Why upstream is safe:** their KGs are 10⁴–10⁵ nodes; ours are ~19. Same
$\alpha$, completely different regime.
- **Verify:** log $\sum_u m_u$, $\sum_u T_u$, and the fraction of saturated
nodes at iteration 1 vs 500, at $\alpha \in \{1.5,\, 10\}$.

### B2. `epsilon` is 1e-4, upstream production runs 0.01

- [ ] Fix

- **Ours:** `vl_base.py:27` · **Upstream:** `config.py:54`
- Ours is stricter, making the early-stop check at `flow_diffusion.py` (~line 291)
essentially never fire. Low impact on its own; fix alongside B1.

### B3. `qa_warm_walk` was not ported

- [ ] Decide whether to add

- **Upstream:** `graph_adapter.py:122`, used at lines 193–206 — when true, the
warm-start random walk spreads proportional to $w'$ instead of uniformly.
- **Ours:** the warm-start loop (`initialize`, ~lines 212–224) always spreads
uniformly. One of five query-awareness injection points, missing.

### B4. Seed-weight floor was not ported

- [ ] Decide whether to add

- **Upstream:** `graph_adapter.py:158-159` — `sw[sw < 1e-5] = 0.0` before
normalising. Prevents near-zero seeds from diluting the distribution.
- **Ours:** `initialize` normalises without a floor. Minor.

### B5. Hybrid coefficients are hardcoded

- [ ] Decide whether to expose

- **Upstream:** `hybrid_a=1.0`, `hybrid_b=0.5` as constructor params.
- **Ours:** hardcoded in `get_edge_weight` (~line 154). Only matters if you
want to tune the hybrid scheme.

---

## Part C — Plumbing (these block diagnosis)

### C1. `batch_push` and all `qa_*` modulators are unreachable

- [ ] Thread through `VideoQAConfig` → `VLQueryParam` → CLI

- **Where:** `video_qa/runner.py:38-43` (`VideoQAConfig` fields) and
`runner.py:112-123` (`VLQueryParam` construction).
- `VideoQAConfig` carries only alpha/epsilon/step_size/max_iterations/
weight_scheme/sim_mode. So `batch_push`, `qafd_sink_gamma`,
`qafd_warm_delta`, `qafd_accum_gamma`, `qafd_post_lambda` are **permanently
at their defaults** (False / 0.0) no matter what you pass on the CLI.
- **Why it matters:** that's four of the five query-awareness injection points
hard-off, plus the push policy.

### C2. `--qafd-weight-scheme` can't express "off"

- [ ] Add `"none"` to the choices

- **Where:** `cli.py:41` restricts to `["multiply", "add", "hybrid"]`, but the
off-switch in `qafd_retriever.py:139` tests for `"none"`.
- **Why it matters:** this is *the* ablation — "does query-awareness do anything
at all?" — and it currently cannot be run from the CLI.

### C3. `batch_push=False` makes edge weights barely effective

- [ ] Understand, then decide the default

- **Where:** `qafd_retriever.py:287` `random.choice(excess_nodes)` — one node
per iteration.
- **Upstream says so explicitly**, `graph_adapter.py:354-356`: batch push
"makes edge weights effective because each iteration touches all excess
nodes' edges, not just one random node's."
- Both implementations default to `False`. Worth testing `True` given our
graphs are tiny (19 nodes — batch push is cheap here, unlike on a 10⁵-node KG).

---

## Part D — Integration bugs (ours, not upstream's)

### D1. VDB scores are overwritten by diffusion mass

- [ ] Fix or justify

- **Where:** `vl_ops.py:232-234`
  ```python
  scores_by_id.update(qafd_scores)          # clobbers VDB distances
  selected_node_ids = list(qafd_scores.keys())   # = every node
  ```
- Final ranking is 100% diffusion mass, 0% vector similarity. Similarity enters
only indirectly, through $s_u$ and $\sigma_u$.
- **Question to answer:** is that intended? Upstream keeps entity and passage
score streams separate and reads out only at passage nodes. A blend
(e.g. $x_u$ rescaled and combined with the VDB score) may be the right call —
but it should be a decision, not a side effect.

### D2. Score dict isn't filtered to positive

- [ ] Fix

- **Ours:** `qafd_retriever.py:304` returns `dict(self.x)`.
- **Upstream:** `flow_diffusion.py:272` returns `{n: v for n, v in self.x.items() if v > 0}`.
- Combined with warm-start giving every reachable node nonzero $x$, our
candidate set is effectively all of $V$ — the cap at `max_segments_for_qa`
does all the real selecting.

### D3. Structural weights carry almost no information

- [ ] Understand; decide whether to change indexing

- **Where:** `vl_ops.py:277` (`temporal_adjacent` weight ≡ 1.0) and
`vl_ops.py:339` (`semantic_sim` weight = cosine, but only created above
`--sim-threshold`, default `0.9` at `cli.py:34`).
- So $w \in \{1.0\} \cup [0.9, 1.0]$ — near-constant graph-wide. There's very
little for $w'$ to modulate, and diffusion sits close to an unweighted random
walk.
- **Options:** lower the threshold to widen the range, or rescale semantic
weights to spread $[0.9, 1.0] \to [0, 1]$.

### D4. `sim_mode="normalized"` compresses relevance

- [ ] Test `relu` / `relu_sq`

- **Where:** `qafd_retriever.py:19-50`.
- $(c+1)/2$ maps realistic cosines $[0.2, 0.6]$ to $[0.6, 0.8]$ — a 1.3× spread,
vs 3× for `relu` and 9× for `relu_sq`. Multiplies with D3 to make $w'$ nearly
uniform.

---

## Part E — Diagnostics (no GT, no VLM re-runs needed)

Run these **after** B1 + C1 + C2, so you're measuring the intended algorithm.
Each checks a claim the design doc makes.

`runner.py:153-155` already writes `seed_node_ids`, `selected_node_ids`, and
`retrieval_trace` into every result row, so E1 and E3 are computable from runs
you already have.

### E1. Query-invariance — *is it query-aware at all?*

- [ ] Run

- For one video, take all its questions; compute mean pairwise Jaccard of
`selected_node_ids`. Compare against `naive` mode as the floor.
- **Fails if:** Jaccard ≈ 1.0 — same segments returned regardless of question.

### E2. Ablation-invariance — *does the query term do anything?*

- [ ] Run (needs C2)

- `weight_scheme="multiply"` vs `"none"`, everything else fixed.
- **Fails if:** output sets are identical — the entire contribution is inert.

### E3. Hop-reach — *is it multi-hop?*

- [ ] Run

- Graph distance from each selected node to its nearest seed.
- **Fails if:** all hops ∈ {0, 1} — it's `local` mode with extra steps, and the
multi-hop claim has no support.

### E4. Degree correlation — *is it just centrality?*

- [ ] Run

- Spearman $\rho$ between final $x_u$ and node degree, across queries.
- **Fails if:** $\rho$ is high and stable across different queries — confirms
the B1 saturation prediction.

### E5. Convergence audit

- [ ] Run

- Log iterations used, pushes, residual excess, and fraction of saturated nodes.
- **Expect:** never converges (that's structural, $\rho = \alpha > 1$). What
you're actually looking for is how the residual and saturation fraction differ
between $\alpha = 1.5$ and $\alpha = 10$.

---

## Part F — The design question (not a bug)

### F1. The substrate has no layer to hop through

- [ ] Decide

Upstream diffuses over a **bipartite entity–passage KG**:

- Seeds come from LLM-extracted OpenIE triples, reranked by an LLM
(`retriever.py:255-310`), plus dense passage hits scaled by
`passage_node_weight = 0.05`.
- Scores are read off **passage nodes only** (`graph_adapter.py:461`) — entity
nodes are *conduits*, discarded from the output.
- Their multi-hop means: reach a passage via entities it shares with the seeds.

Our graph has one node type. `semantic_sim` edges encode segment–segment cosine
similarity — **the same signal the vector search already used**. There is no
intermediate node type to route through.

**The question:** can flow diffusion express genuine multi-hop retrieval on a
single-layer similarity graph, or does the event graph need an object/entity
layer to give diffusion something to traverse?

This is RQ1 from `[proj_description.md](../proj_notes/proj_description.md)`, not
a tuning issue. Don't decide it until B–E are cleared — right now we can't tell
whether QAFD underperforms because the substrate is wrong or because
$\alpha$ is 7× too high.

### F2. Benchmark selection

- [ ] Decide

See `[benchmarks.md](../proj_notes/benchmarks.md)`, which argues **LongVidSearch**
over CG-Bench for the multi-hop claim (Hop-2/3/4 with *necessary* evidence
clips; CG-Bench is typically one clue interval, which tests localisation not
traversal). Gated on F1 — no benchmark will be informative until the mechanism
is known to work.

---

## Suggested order

```
A1 → A2 → A3          orientation
  ↓
B1 → B2               parameter fixes (B1 is the big one)
  ↓
C1 → C2               plumbing, unblocks the ablation
  ↓
E1 → E3 → E4          cheapest diagnostics, no re-runs needed
  ↓
E2 → E5               need C2 / instrumentation
  ↓
D1 → D2 → D3 → D4     integration, informed by what E showed
  ↓
B3 → B4 → C3          remaining knobs, tune against E's metrics
  ↓
F1 → F2               the actual research decision
```

**Rationale:** B1 is one line and may change behaviour qualitatively, so it
comes before any measurement. C1/C2 are pure plumbing but E2 is impossible
without them. The E block is cheap and tells you which of D3/D4 is worth
chasing. F waits until you can attribute failure to the substrate rather than
to configuration.