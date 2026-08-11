# Query-Aware Flow Diffusion (QAFD) Retrieval Design

## Overview

The Query-Aware Flow Diffusion (QAFD) retriever (`graphrag/vl_graphrag/qafd_retriever.py`) introduces a novel dynamic retrieval mechanism over video event graphs. Unlike the static `local` 1-hop expansion mode in traditional VLGraphRAG, QAFD modulates graph traversal dynamically based on the semantic relevance of nodes to the user's query.

It leverages a Push-Relabel flow diffusion algorithm to propagate "importance mass" from seed nodes (initially retrieved via vector search) through the temporal and semantic edges of the NetworkX event graph. 

## Key Concepts

### 1. Dynamic Edge Weighting
The core innovation of QAFD is modifying static structural edge weights $w(u, v)$ with a query-aware factor:
$$w'(u, v) = w(u, v) \times f(\text{sim}(u, q), \text{sim}(v, q))$$

Where:
- $q$ is the query embedding.
- $u, v$ are node embeddings.
- $\text{sim}(\cdot, \cdot)$ is the cosine similarity (configurable as `normalized`, `relu`, or `relu_sq`).
- $f$ is the weighting scheme (e.g., `multiply`, `add`, `hybrid`).

This ensures that flow (and therefore relevance) preferentially travels along paths that are semantically related to the user's query.

### 2. Flow Diffusion Algorithm
The retrieval algorithm works as follows:
1. **Initialization:** 
   - Seed nodes (from initial VDB retrieval) receive an initial mass proportional to their seed scores.
   - Sink capacities are assigned based on node degree (optionally modulated by query relevance).
   - A lazy random walk "warm-starts" the node scores ($x$).
2. **Push-Relabel Operations:**
   - Nodes with excess mass (mass > sink capacity) "push" this excess to their neighbors.
   - The amount of mass pushed is proportional to the query-aware edge weight $w'(u, v)$.
3. **Convergence:**
   - The diffusion continues until no node has excess mass greater than a small threshold ($\epsilon$) or until `max_iterations` is reached.

### 3. Query-Aware Modulators
QAFD provides several hyperparameters to inject query-awareness at different stages of the diffusion process:
- **`qa_sink_gamma`**: Modulates the sink capacity of nodes based on their similarity to the query.
- **`qa_warm_delta`**: Modulates the warm-start probabilities.
- **`qa_accum_gamma`**: Modulates the mass accumulation during push steps.
- **`qa_post_lambda`**: Applies a final score boost to retrieved nodes based on query similarity.

## Integration with VLGraphRAG

The `NetworkXQAFDRetriever` is integrated into the `VLQueryParam` under the `mode="qafd"` flag. 
When triggered, `vl_ops.py` bypasses the standard 1-hop expansion and instead instantiates the QAFD retriever, passing in:
- The `segments_relation_graph` (NetworkX graph).
- The `seed_node_ids` and their initial vector search scores.
- The `node_embeddings` (from the VDB).
- The `query_embedding`.

The resulting scores from QAFD are then used to rank and cap the final retrieved segments for the VideoQA prompt.
