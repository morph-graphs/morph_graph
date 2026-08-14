# Query-Aware Flow Diffusion (QAFD) Retrieval Design

Implementation: `graphrag/vl_graphrag/qafd_retriever.py` (`NetworkXQAFDRetriever`)
Call site: `graphrag/vl_graphrag/vl_ops.py::vl_query`, under `mode="qafd"`
Upstream reference: [QAFD-RAG](https://arxiv.org/abs/2605.18775), `src/passage_entity/graph_adapter.py::IGraphQAFD`

Companion: [qafd_worklist.md](./qafd_worklist.md) — the "what's wrong and why" checklist.

## Overview

QAFD retrieves a *connected, query-relevant* set of event nodes rather than a
flat top-k by similarity. It injects "importance mass" at seed nodes (from
vector search) and diffuses it through the event graph via push-relabel flow
diffusion, where **edge traversal weights are modulated by query relevance**.
Nodes accumulate an importance score proportional to the flow passing through
them; the top-scoring nodes become the retrieved segments.

This contrasts with the two existing modes:

| Mode | Candidate set | Ranking signal |
| --- | --- | --- |
| `naive` | VDB top-k | query–node cosine similarity |
| `local` | VDB top-k + 1-hop neighbours | query–node cosine similarity |
| `qafd` | all nodes reached by diffusion | accumulated flow $x_u$ |

---

## Notation

| Symbol | Meaning |
| --- | --- |
| $G=(V,E)$ | event graph (undirected; `segments_relation_graph`) |
| $N(u)$, $d_u$ | neighbours of $u$, degree of $u$ |
| $q$ | query embedding |
| $e_u$ | embedding of node $u$ (from the VDB matrix) |
| $\sigma_u$ | query relevance of node $u$, $\sigma_u \in [0,1]$ |
| $w(u,v)$ | structural edge weight (stored on the edge) |
| $w'(u,v)$ | query-aware edge weight |
| $S \subseteq V$ | seed nodes (VDB top-k), with scores $\text{score}_u$ |
| $s_u$ | normalised seed weight, $\sum_u s_u = 1$ |
| $m_u$ | mass currently held at $u$ |
| $T_u$ | sink capacity of $u$ (how much mass $u$ can absorb) |
| $r_u$ | excess at $u$, $r_u = m_u - T_u$ |
| $x_u$ | accumulated importance score — **the retrieval output** |

Greek hyperparameters: $\alpha$ (mass scale), $\eta$ (step size),
$\epsilon$ (convergence tolerance), $\gamma_{\text{sink}}$, $\delta_{\text{warm}}$,
$\gamma_{\text{acc}}$, $\lambda_{\text{post}}$ (query-aware modulators).

---

## 1. Query relevance

Let $c_u$ be the raw cosine between node and query:

$$
c_u \;=\; \frac{\langle e_u,\, q \rangle}{\lVert e_u \rVert \, \lVert q \rVert} \;\in\; [-1, 1]
$$

`sim_mode` selects the contrast function applied to $c_u$:

$$
\sigma_u \;=\;
\begin{cases}
\max\!\left(0,\; \dfrac{c_u + 1}{2}\right) & \texttt{normalized} \\[2ex]
\max(0,\; c_u) & \texttt{relu} \\[1ex]
\max(0,\; c_u)^2 & \texttt{relu\_sq}
\end{cases}
$$

All three map to $[0,1]$, but their **dynamic range differs sharply**. For
embeddings whose realistic cosines lie in $c \in [0.2, 0.6]$:

| `sim_mode` | $\sigma$ range | spread |
| --- | --- | --- |
| `normalized` | $[0.60,\, 0.80]$ | $1.3\times$ |
| `relu` | $[0.20,\, 0.60]$ | $3\times$ |
| `relu_sq` | $[0.04,\, 0.36]$ | $9\times$ |

`normalized` compresses relevance into a narrow band near $0.7$, which
substantially weakens every downstream query-aware term. It is the default for
parity with upstream, not because it is the best choice here.

## 2. Query-aware edge weighting

The core modulation. `weight_scheme` selects $f$ in
$w'(u,v) = f\big(w(u,v),\, \sigma_u,\, \sigma_v\big)$:

$$
w'(u,v) \;=\;
\begin{cases}
w(u,v)\,\sigma_u\,\sigma_v & \texttt{multiply} \\[1ex]
\dfrac{w(u,v) + \sigma_u + \sigma_v}{3} & \texttt{add} \\[2ex]
w(u,v)\left(a + b\,\dfrac{\sigma_u + \sigma_v}{2}\right) & \texttt{hybrid},\quad a = 1,\; b = \tfrac{1}{2} \\[2ex]
w(u,v) & \texttt{none}
\end{cases}
$$

Guards: $w'(u,v) = 0$ whenever $w(u,v) \le 0$, and $w' = w$ whenever embeddings
or the query embedding are unavailable. Weights are cached per unordered pair.

**Where $w(u,v)$ comes from** (`vl_ops.py`), which bounds how much this term can
do:

- `temporal_adjacent` edges: $w \equiv 1.0$, constant.
- `semantic_sim` edges: $w =$ cosine similarity, but only created above
  `semantic_similarity_threshold` (CLI default $0.9$), so $w \in [0.9, 1.0]$.

So $w$ is near-constant across the whole graph, and under `multiply` +
`normalized` the modulation factor $\sigma_u\sigma_v$ spans roughly
$[0.36, 0.64]$. The resulting $w'$ is close to uniform — the diffusion is
therefore close to an unweighted random walk unless `sim_mode` is sharpened or
the structural weights are made more discriminative.

## 3. Initialisation

**Seed weights.** From the VDB top-k scores, clamped and normalised:

$$
s_u \;=\; \frac{\max(0,\, \text{score}_u)}{\sum_{v \in V} \max(0,\, \text{score}_v)},
\qquad \sum_{u} s_u = 1
$$

If all scores are zero, $s_u = 1/|V|$ (uniform fallback). Non-seed nodes have
$\text{score}_u = 0$, hence $s_u = 0$.

**Sink capacities.** Proportional to degree, then normalised to a fixed budget
of $10$:

$$
T_u^{(0)} = \begin{cases} \max(d_u,\, 1) & \texttt{use\_node\_degree} \\ 1 & \text{otherwise} \end{cases}
\qquad\Longrightarrow\qquad
T_u \;=\; 10 \cdot \frac{T_u^{(0)}}{\sum_{v} T_v^{(0)}}
$$

so that $\sum_u T_u = 10$ exactly. Optionally biased toward query-relevant nodes:

$$
T_u \;\leftarrow\; T_u \left(1 + \gamma_{\text{sink}}\, \sigma_u\right)
$$

Write $T = \sum_u T_u$ for the (post-bias) total.

**Mass injection.** All mass enters at the seeds:

$$
m_u \;=\; \alpha\, T\, s_u
\qquad\Longrightarrow\qquad
\sum_u m_u \;=\; \alpha\, T
$$

### The saturation ratio

Combining the two lines above gives the single most important derived quantity:

$$
\boxed{\;\rho \;=\; \frac{\sum_u m_u}{\sum_u T_u} \;=\; \alpha\;}
$$

**$\alpha$ *is* the ratio of injected mass to total absorbing capacity.** Two
consequences follow directly:

1. For any $\alpha > 1$ the system **cannot** reach the "no node has excess"
   state — $(\alpha - 1)$ units of mass per unit of sink circulate
   indefinitely. Non-convergence is structural, not a failure; the residual
   excess is what defines the retrieved cluster. The `max_iterations` cap is
   the *de facto* stopping rule.
2. $\alpha$ controls **how far the flood spreads**. Small $\alpha$ keeps flow
   concentrated near the seeds, so $x$ reflects seed proximity. Large $\alpha$
   saturates every node to its capacity, after which $x_u$ grows by
   $\eta\, r_u / W_u$ everywhere — i.e. **by degree**, with the seed signal
   washed out. The retrieval then degenerates toward a query-independent
   centrality ranking.

Because the budget $\sum_u T_u$ is pinned at $10$ regardless of $|V|$, the
per-node capacity $T_u \approx 10/|V|$ shrinks as the graph grows. A given
$\alpha$ is therefore **far more saturating on a small graph**: on a
$19$-node event graph, $\alpha = 10$ injects $100$ units against $\approx 0.53$
capacity per node. See [Parameters](#parameters) for the upstream/local divergence.

**Warm start.** $x$ is seeded by a $k$-step lazy random walk
($k =$ `qa_warm_steps`), optionally biased toward query-relevant seeds:

$$
x^{(0)}_u \;=\; \frac{s_u\left(1 + \delta_{\text{warm}}\, \sigma_u\right)}{\sum_{v} s_v \left(1 + \delta_{\text{warm}}\, \sigma_v\right)}
\qquad \left(x^{(0)} = s \ \text{ when } \delta_{\text{warm}} = 0\right)
$$

$$
x^{(t+1)}_u \;=\; \frac{1}{2}\left(s_u \;+\; \sum_{v \in N(u)} \frac{x^{(t)}_v}{d_v}\right)
$$

The $\tfrac{1}{2}(s_u + \cdot)$ form re-anchors each step to the seed
distribution, keeping $x$ concentrated rather than drifting to the stationary
distribution.

## 4. The push operation

For a node $u$ with excess $r_u = m_u - T_u > 0$, define the query-aware and
structural neighbourhood sums:

$$
W'_u \;=\; \sum_{v \in N(u)} w'(u,v),
\qquad
W_u \;=\; \sum_{v \in N(u)} w(u,v)
$$

The push then does three things, in order:

**(a) Accumulate** importance at $u$ — scaled by the *structural* sum $W_u$:

$$
x_u \;\mathrel{+}=\; \eta\, \frac{r_u}{W_u + 10^{-8}} \left(1 + \gamma_{\text{acc}}\, \sigma_u\right)
$$

**(b) Absorb** what the sink can hold, zeroing the excess:

$$
m_u \;\leftarrow\; T_u
$$

**(c) Route** the excess to neighbours — scaled by the *query-aware* sum $W'_u$:

$$
m_v \;\mathrel{+}=\; r_u\, \frac{w'(u,v)}{W'_u + 10^{-8}}
\qquad \forall\, v \in N(u)
$$

> **Accumulation and routing are deliberately decoupled.** $x$ accumulates by
> structural degree; mass routes by query-aware weights. Inherited from upstream
> (`graph_adapter.py::_push`), the rationale being that query modulation should
> *steer* flow without also penalising the accumulation rate at query-relevant
> nodes — under `multiply`, $W'_u \le W_u$, so sharing a denominator would
> inflate $x$ exactly where $\sigma$ is low.

If $W'_u = 0$ (all incident query-aware weights vanish), the push is skipped and
$u$ retains its excess. $W_u = 0$ falls back to $W_u \leftarrow W'_u$.

## 5. Diffusion loop

Let $X^{(t)} = \{\,u \in V : m_u > T_u + \epsilon\,\}$ be the excess set at
iteration $t$. Each iteration:

1. If $X^{(t)} = \varnothing$ → **converged**, stop.
2. Push. `batch_push` selects the policy:
   - `False` (default): push one $u \sim \mathrm{Uniform}(X^{(t)})$.
   - `True`: push **every** $u \in X^{(t)}$.
3. Every $10$ iterations, stop early if total residual excess is negligible:
   $\sum_u \max(0,\, m_u - T_u) < \epsilon$.
4. Stop unconditionally at `max_iterations`.

> **The push policy determines whether edge weights matter.** With
> `batch_push=False`, one iteration touches the edges of a *single random* node,
> so `max_iterations` pushes are spread thinly and $w'$ has few opportunities to
> steer anything. Upstream documents this explicitly: batch push "makes edge
> weights effective because each iteration touches all excess nodes' edges, not
> just one random node's" (`graph_adapter.py::run`). Both implementations
> nonetheless default to `False`.

## 6. Post-diffusion reranking and selection

An optional final boost by query relevance:

$$
\hat{x}_u \;=\; x_u \left(1 + \lambda_{\text{post}}\, \sigma_u\right)
$$

`vl_query` then ranks all scored nodes by $\hat{x}$ and caps at
`max_segments_for_qa`:

$$
\text{selected} \;=\; \operatorname*{arg\,top-}_{u}{}^{K} \; \hat{x}_u,
\qquad K = \texttt{max\_segments\_for\_qa}
$$

Note that the returned score map **replaces** the VDB distances for seed nodes
(`scores_by_id.update(qafd_scores)`), so the final ordering is determined
entirely by diffusion mass — vector similarity enters only through $s_u$ and
$\sigma_u$, never directly as a ranking term.

---

## Query-aware modulators — summary

Query relevance $\sigma$ enters at five separate points. All but the edge
weighting default to *off*:

| Point | Parameter | Effect | Default |
| --- | --- | --- | --- |
| Edge weights | `weight_scheme` | $w'(u,v)$ — routes mass along relevant paths | `multiply` (**on**) |
| Sink capacity | $\gamma_{\text{sink}}$ | relevant nodes absorb more mass | `0.0` (off) |
| Warm start | $\delta_{\text{warm}}$ | biases initial $x$ toward relevant seeds | `0.0` (off) |
| Accumulation | $\gamma_{\text{acc}}$ | relevant nodes accrue $x$ faster | `0.0` (off) |
| Post-rerank | $\lambda_{\text{post}}$ | final multiplicative boost | `0.0` (off) |

Setting `weight_scheme="none"` with all modulators at $0$ recovers plain
(query-independent) flow diffusion — the natural ablation for measuring what
query-awareness contributes.

## Parameters

Defaults are defined in `VLQueryParam` (`vl_base.py`) and threaded through
`VideoQAConfig` (`video_qa/runner.py`).

| Parameter | Symbol | Ours | Upstream production[^1] | Notes |
| --- | --- | --- | --- | --- |
| `qafd_alpha` | $\alpha$ | `10.0` | **`1.5`** | saturation ratio $\rho = \alpha$ |
| `qafd_epsilon` | $\epsilon$ | `1e-4` | `0.01` | excess tolerance |
| `qafd_max_iterations` | — | `500` | `500` | de facto stopping rule |
| `qafd_step_size` | $\eta$ | `0.2` | `0.2` | accumulation rate |
| `qafd_weight_scheme` | $f$ | `multiply` | `multiply` | |
| `sim_mode` | — | `normalized` | `normalized` | see §1 on dynamic range |
| `batch_push` | — | `False` | `False` | see §5 |
| `qafd_sink_gamma` | $\gamma_{\text{sink}}$ | `0.0` | `0.0` | |
| `qafd_warm_delta` | $\delta_{\text{warm}}$ | `0.0` | `0.0` | |
| `qafd_warm_steps` | $k$ | `2` | `2` | |
| `qafd_accum_gamma` | $\gamma_{\text{acc}}$ | `0.0` | `0.0` | |
| `qafd_post_lambda` | $\lambda_{\text{post}}$ | `0.0` | `0.0` | |

[^1]: `QAFD-RAG/src/passage_entity/config.py::PassageEntityConfig`, the values
actually passed at `retriever.py::graph_search_with_fact_entities`. Note these
differ from the *class signature* defaults in `graph_adapter.py`
($\alpha = 10.0$, $\epsilon = 10^{-6}$), which production always overrides.

> **⚠ Open discrepancy — `alpha`.** Our $\alpha = 10.0$ matches upstream's class
> signature default, not the config that upstream actually runs ($1.5$). By
> $\rho = \alpha$ this is a $\approx 7\times$ higher saturation ratio, applied to
> graphs $3$–$4$ orders of magnitude smaller than upstream's entity KGs. Per §3,
> this is the regime where $x$ degenerates toward degree ranking. Unresolved —
> tracked as B1 in [qafd_worklist.md](./qafd_worklist.md).

Not ported from upstream: `qa_warm_walk` (use $w'$ rather than uniform spread
during the warm-start walk), the seed-weight floor $s_u < 10^{-5} \mapsto 0$,
and the configurable hybrid coefficients $a, b$ (hardcoded to $1, \tfrac{1}{2}$).

---

## Integration with VLGraphRAG

`vl_query` (`vl_ops.py`) dispatches on `param.mode == "qafd"`, bypassing the
1-hop `expand_node_ids` path:

```
query
  ├─ vdb.query(top_k)                  → seed_node_ids, scores  → s_u
  ├─ vdb.embedding_func([query])       → query_embedding        → q
  ├─ get_node_embeddings_from_vdb(vdb) → node_embeddings        → e_u
  └─ segments_relation_graph._graph    → nx.Graph               → G
        │
        └─ NetworkXQAFDRetriever(...).retrieve()  → {node_id: x̂_u}
              │
              └─ rank_and_cap_node_ids(..., max_segments_for_qa)
                    → segments_kv.get_by_ids → caption_context / stitched video
```

The retriever returns scores for **every** node the diffusion touched (warm
start gives all reachable nodes $x_u > 0$), so the candidate set is effectively
all of $V$ — the cap at `max_segments_for_qa` does the actual selection.

### Substrate caveat

Upstream diffuses over a **bipartite entity–passage KG**: passages connect
through a shared-entity layer, entity nodes act as conduits, and scores are read
off passage nodes only. Multi-hop retrieval there means *reaching a passage via
entities it shares with the seeds*.

Our graph has no entity layer — nodes are segments, and `semantic_sim` edges
encode segment–segment cosine similarity, the same signal vector search already
used. There is no intermediate node type to hop *through*. Whether flow
diffusion can express genuine multi-hop retrieval on this substrate, or whether
the event graph needs an object/entity layer, is an open design question rather
than a tuning issue.

## Related

- [QAFD issue worklist](./qafd_worklist.md) — prioritised audit checklist
- [GraphRAG core functionality](../proj_docs/graphrag.md) — indexing, storage, other retrieval modes
- [Benchmarks shortlist](../proj_notes/benchmarks.md) — evidence-grounded multi-hop eval sets
- [Project description](../proj_notes/proj_description.md) — RQ1: does relational traversal beat similarity-only retrieval?
