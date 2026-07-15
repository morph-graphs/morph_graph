# Streaming & Long Video Understanding Papers

Organized by **problem setting** (what horizon / benchmarks) and **method** (how memory / reasoning is built).

---

## Problem Settings

### Medium / Long Videos (VideoMME, MLVU, …)

Focuses on minutes-to-hours videos and multi-hop / long-context QA. Not primarily about real-time / online streaming interaction (StreamingBench / OVO-style).

| Title | Identifier | Venue | Summary |
| --- | --- | --- | --- |
| VideoDetective: Clue Hunting via both Extrinsic Query and Intrinsic Relevance for Long Video Understanding | [2603.22285](https://arxiv.org/abs/2603.22285) | Preprint | Builds a spatio-temporal affinity graph and runs Hypothesis–Verification–Refinement with graph diffusion so sparse segment observations recover a global query-relevance map (“see less but know more”). |
| Video-o3: Native Interleaved Clue Seeking for Long Video Multi-Hop Reasoning | [2601.23224](https://arxiv.org/abs/2601.23224) | Preprint | Iteratively discovers clues via interleaved tool calls (e.g. VideoCrop), with task-decoupled attention masking and trajectory-guided RL; strong on MLVU / Video-Holmes. |
| Vgent: Graph-based Retrieval-Reasoning-Augmented Generation For Long Video Understanding | [2510.14032](https://arxiv.org/abs/2510.14032) | Preprint | Offline clip graph + structured verification between retrieval and generation to cut RAG noise on long videos (gains on MLVU). |
| GraphThinker: Reinforcing Temporally Grounded Video Reasoning with Event Graph Thinking | [2602.17555](https://arxiv.org/abs/2602.17555) | Preprint | Builds an event-based video scene graph (intra-/inter-event relations) as intermediate “graph thinking” and RL-finetunes with a visual attention reward to reduce hallucinations. |
| Graph-to-Frame RAG: Visual-Space Knowledge Fusion for Training-Free and Auditable Video Reasoning | [2604.04372](https://arxiv.org/abs/2604.04372) | CVPR 2026 | Offline problem-agnostic video KG; at query time retrieves a minimal subgraph and renders it as one reasoning frame fused in visual space. |

### Ultra-Long Videos (EgoLifeQA, days-long / personalized egocentric)

Focuses on day-scale or life-log horizons and personalization over long egocentric history.

| Title | Identifier | Venue | Summary |
| --- | --- | --- | --- |
| EgoSelf: From Memory to Personalized Egocentric Assistant | [2604.19564](https://arxiv.org/abs/2604.19564) | Preprint | Builds a graph-based interaction memory over long egocentric history and trains personalization by predicting future interactions from user-specific graph profiles. |

---

## Methods

### Streaming Video Understanding with Graph Memory

**Explicit graph memory** (or closely related structured event graphs updated online).

| Title | Identifier | Venue | Graph? | Summary |
| --- | --- | --- | --- | --- |
| What Should a Streaming Video Model Remember? (SelectStream) | [2606.16353](https://arxiv.org/abs/2606.16353) | Preprint | **Yes** — query-conditioned graph reasoning over a fixed-capacity latent memory graph | Budgeted online latent evidence allocation: surprise-driven windowing, consolidation, and query-conditioned graph retrieval injected as latent tokens (StreamingBench / OVO-Bench). |
| EgoSelf: From Memory to Personalized Egocentric Assistant | [2604.19564](https://arxiv.org/abs/2604.19564) | Preprint | **Yes** — graph-based interaction memory | Long-horizon egocentric graph of events/entities for personalized future-interaction prediction. |
| EventMemAgent: Hierarchical Event-Centric Memory for Online Video Understanding with Adaptive Tool Use | [2602.15329](https://arxiv.org/abs/2602.15329) | Preprint | Event hierarchy (STM/LTM); tool-augmented retrieval over events | Online agent with event-boundary STM + event-archived LTM and tools (`search_memory`, OCR, detect); evaluated on StreamingBench / OVO-Bench. |

### Streaming Video Understanding with Thinking

Interleaves / amortizes reasoning (or test-time adaptation) while the stream advances.

| Title | Identifier | Venue | Summary |
| --- | --- | --- | --- |
| Video Streaming Thinking: VideoLLMs Can Watch and Think Simultaneously | [2603.12262](https://arxiv.org/abs/2603.12262) | ECCV 2026 | Interleaves reasoning with streaming video playback so intermediate thoughts amortize CoT cost and enable low-latency query responses. |
| Think While Watching: Online Streaming Segment-Level Memory for Multi-Turn Video Reasoning in Multimodal Large Language Models | [2603.11896](https://arxiv.org/abs/2603.11896) | ECCV 2026 | Maintains segment-level streaming memory with causal masks so MLLMs can overlap watching and thinking across multi-turn interaction. |
| Thinking in Streaming Video | [2603.12938](https://arxiv.org/abs/2603.12938) | ECCV 2026 | ThinkStream’s Watch–Think–Speak paradigm with reasoning-compressed streaming memory (RCSM) and streaming RL for incremental online video reasoning. |
| Spatial-TTT: Streaming Visual-based Spatial Intelligence with Test-Time Training | [2603.12255](https://arxiv.org/abs/2603.12255) | Preprint | Uses TTT fast weights as compact non-linear memory (with 3D spatiotemporal conv) to accumulate spatial evidence from unbounded streams. |

### Streaming Video Understanding with KV-Cache Memory

Bounded/reusable KV cache as memory — no explicit relational structure; included as a streaming efficiency baseline.

| Title | Identifier | Venue | Summary |
| --- | --- | --- | --- |
| HERMES: KV Cache as Hierarchical Memory for Efficient Streaming Video Understanding | [2601.14724](https://arxiv.org/abs/2601.14724) | ACL 2026 | Training-free; treats the KV cache as multi-granularity hierarchical memory and reuses a compact KV cache for real-time streaming — no query-time auxiliary compute (~10× faster TTFT), up to ~68% token reduction with comparable/better accuracy. |

### Offline Video Understanding with Graph Memory

Graph is built (or fully available) offline / at inference over the full video, then used for retrieval or intermediate thinking — not maintained as an online stream state.

| Title | Identifier | Venue | Summary |
| --- | --- | --- | --- |
| Graph-to-Frame RAG: Visual-Space Knowledge Fusion for Training-Free and Auditable Video Reasoning | [2604.04372](https://arxiv.org/abs/2604.04372) | CVPR 2026 | Offline video KG → online minimal subgraph rendered as a single visual reasoning frame (training-free, auditable). |
| Vgent: Graph-based Retrieval-Reasoning-Augmented Generation For Long Video Understanding | [2510.14032](https://arxiv.org/abs/2510.14032) | Preprint | Offline semantic clip graph + intermediate structured verification to denoise RAG for long-video QA (e.g. MLVU). |
| VideoDetective: Clue Hunting via both Extrinsic Query and Intrinsic Relevance for Long Video Understanding | [2603.22285](https://arxiv.org/abs/2603.22285) | Preprint | Spatio-temporal affinity graph + sparse Observation + graph diffusion for global query relevance under limited context. |
| GraphThinker: Reinforcing Temporally Grounded Video Reasoning with Event Graph Thinking | [2602.17555](https://arxiv.org/abs/2602.17555) | Preprint | Event video scene graphs as intermediate thinking + RL visual-attention reward for grounded, temporally consistent reasoning. |

---

## Quick Cross-Walk

| Paper | Problem setting | Method family |
| --- | --- | --- |
| SelectStream | Streaming (medium/long online) | Streaming + **graph** memory |
| EventMemAgent | Streaming (online) | Streaming + structured event memory |
| EgoSelf | Ultra-long egocentric | Streaming/long-horizon + **graph** memory |
| VST / Think While Watching / ThinkStream | Streaming | Streaming + thinking |
| Spatial-TTT | Streaming spatial | Streaming + TTT memory |
| HERMES | Streaming (online) | Streaming + KV-cache memory |
| Video-o3 | Medium/long (MLVU-style) | Offline interleaved clue-seeking / thinking |
| G2F-RAG / Vgent / VideoDetective / GraphThinker | Medium/long | Offline + **graph** memory |
