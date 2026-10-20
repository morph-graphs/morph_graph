# Morph Graphs: Training Notes

Living brainstorm notes. Updated as we discuss. Keep entries terse; link to code/papers rather than restating them.

How this file is organized: secs. 1-2 say what we are building and where we are. Secs. 3-5 are the design, one section per capability group. Sec. 6 is data. Secs. 7-8 collect every open question and every decision in one place. Secs. 9-10 are reference material (VST pipeline internals, Latent-VC setup and code) that supports the design but should not be read top to bottom. Sec. 11 is papers.

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Status and Next Experiment](#2-status-and-next-experiment)
3. [Cap. 1 Design: Event Segmentation](#3-cap-1-design-event-segmentation)
4. [Cap. 2 Design: Latent Memory](#4-cap-2-design-latent-memory)
5. [Caps. 3-5 Design: Graph Construction, Modulation, Retrieval](#5-caps-3-5-design-graph-construction-modulation-retrieval)
6. [Datasets](#6-datasets)
7. [Open Questions](#7-open-questions)
8. [Decisions Log](#8-decisions-log)
9. [Reference: VST Data-Gen Pipeline](#9-reference-vst-data-gen-pipeline)
10. [Reference: Latent-VC Setup and Codebase](#10-reference-latent-vc-setup-and-codebase)
11. [References](#11-references)

---

## 1. Problem Statement

Design a model that, from a continuous video stream, builds and maintains a *morphing* graph memory and uses it for question answering. Five target capabilities:

| # | Capability | One-liner |
|---|-----------|-----------|
| 1 | Event segmentation | Detect event boundaries at multiple granularities from the stream. |
| 2 | Real-time latent memory | Stream -> compact latent capturing objects, events/actions, scene context. |
| 3 | Real-time graph construction | Latent memory -> explicit graph; nodes = events/concepts, edges link them across time. |
| 4 | Graph topology modulation | Graph rewires as new events arrive, e.g. edges shorten between frequently co-occurring concepts. |
| 5 | Query-aware retrieval | At QA time, the query modulates topology to surface relevant nodes. |

Constraints: everything in 1-4 runs online (single pass, bounded memory). 5 runs at query time.

## 2. Status and Next Experiment

**Phase: prototyping.** Caps. 1 and 2 share one model (secs. 3-4). Latent types are trained independently and offline first; the final model is joint, then online. The boundary baseline (cap. 1) precedes all latent work. Caps. 3-5 are at the idea stage (sec. 5).

**Next experiment (Vatsal): reproduce LVR / Latent-VC pre-training as the untyped warm-up.** Both have code + checkpoints: LVR https://github.com/Penation/lvr (Qwen2.5-VL, LVR-7B), Latent-VC https://github.com/BRZ911/Latent-VC (Qwen3.5-9B, `BRZ911/Latent-VC-9B`). Plan:
1. Probe the released Latent-VC cache tokens on our videos (decode to text, cosine to frame tokens, stability across clips) before any training.
2. Reproduce stage 1 (contrastive cache alignment) to one headline number, skip GRPO.
3. Swap to per-unit emission at Streamo closes.

Add a retention probe their setup lacks: emit cache on clip 1, ask about clip 1 after clips 2-3, measure degradation. **Start with Latent-VC** (video; released checkpoint lets us probe before training; its contrastive alignment is the loss we keep). LVR is the fallback, with a two-day limit on fighting the Qwen3.5-9B environment before switching.

**Where things stand**: Latent-VC repo, models, SFT videos, and eval sets are on the cluster and verified; the conda env is not yet built; the codebase has been read (sec. 10). VST annotations are local; the KG json and evidence paths must be regenerated with VST's `data_gen/` code (sec. 9).

## 3. Cap. 1 Design: Event Segmentation

Direction: **query-free** segmentation, trained as proactive generation in the Streamo / VST style, with MeCo's semantic-first framing (structural tokens + captioning + contrastive grounding) made causal. Caps. 1 and 2 share one model: the caption emitted at a boundary is the memory unit, the hidden state at the structural token is its latent.

**Boundaries typed by kind of shift** (Vatsal; Event Segmentation Theory framing). A boundary is a spatial, goal, or perceptual shift; several can co-occur, so model them as per-second binary heads (or a flag set on one close token), not one categorical token.

| Shift | Latent refreshed | Supervision |
|-------|------------------|-------------|
| Spatial (new place / viewpoint) | context | location changes; context-latent consistency in egocentric video |
| Goal (intention / task step) | event | COIN, YouCook2 steps; ActivityNet, HACS events; **Ego4D Goal-Step** (goal > step > substep tree) |
| Perceptual (shot cut, camera motion, appearance) | none; dense cue only | Kinetics-GEBD |

Multi-granularity comes from goal nesting (goal / step / substep), not duration thresholds. `<silence>` and Streamo's `<standby>` stay as the no-shift states; a `<close:bg>` background unit keeps memory contiguous. Each close is followed by the caption of the unit that ended (VST-style).

**Level assignment**: by goal-tree depth where labels have it (Goal-Step), else by task type (steps -> subgoal level, events -> goal level). Duration histogram no longer needed for levels.

**Hindsight start**: MeCo contrastive grounding restricted to a bounded recent frame buffer with a level-specific window. Close-token state aligns to its own segment's frames. Doubles as the latent-to-frames link for cap. 2.

**Loss** = focal CE over structural tokens (Streamo, handles ~80% silence) + LM loss on captions (VST-style thoughts) + contrastive grounding (MeCo). For VST data mask the structural-token term (clips are equal splits, not semantic boundaries); train captions only.

**Data (query-free subset)**
- Boundaries + captions: Streamo event caption, action caption, real-time narration. Event grounding and time-sensitive QA are query-conditioned; hold for cap. 5.
- Captions only: VST streaming thoughts (75K videos).
- Eval: boundary F1 @ 1-2 s per level on held-out Streamo sources; Kinetics-GEBD for generic boundaries.

**Fallback**: H-Net style learned chunking on frozen features (arXiv 2507.07955), superseded but kept if structural-token timing is too laggy.

## 4. Cap. 2 Design: Latent Memory

### 4.1 What the latents are

**Latent goals** (Vatsal): capture the *gist* of the event / its frames; typed; timestamps optional but useful. Not assumed to be a sufficient substrate on their own. **The VLM generates the latents itself**, as typed output tokens in its stream (as VST emits thoughts), not via an external slot / memory module feeding the LLM. This is the departure from prior typed-memory work.

Three types:
- *Event*: the action / goal segment between clear transitions.
- *Object*: persistent entities incl. people.
- *Context*: the physical setting (room / background scene), not participants or activity. Setting-only targets: strip people/actions from establishing captions, or use same-room consistency. Updates only on location change -> candidate `<close:scene>` token. Graph role: place node that events and objects hang off. Open: static background only, or slowly changing room state too?

Fixed-size block per emission (K latents), as in LVR and Latent-VC. **Per unit, never per frame** (latents are only emitted at closes, so no redundant per-frame latents).

### 4.2 Prototyping recipe: untyped warm-up, then typed, each type offline

**Decision (Vatsal): for prototyping, train each latent type independently and offline.** Not the final recipe; the full model trains jointly, then online on streams (sec. 4.4). Event, context, object as three separate runs with disjoint targets (action-only / setting-only / track identity). Fix latent size and token count per type up front so modules plug into one model later. Joint fine-tune afterwards with the attention ordering context -> object -> event. Boundary baseline still precedes all of this.

**Untyped warm-up, then typed.** Closest paradigm: LVR / CoLVR / Latent Visual Cache / LatentStream: the LLM's own hidden states are the latents, fed back as input embeddings; SFT aligns latents to visual encoder tokens (reconstruction or contrastive), then RL with a grounding reward. Plan:
- Warm-up (~1 week): untyped latent mode, LVR-style reconstruction / contrastive alignment to frame tokens, any video, no labels. Pass/fail on whether the VLM can emit continuous latents and answer from them. Not kept as a representation. This is the experiment in sec. 2.
- Then typed from the start of real training; typing = a type token + a type-specific target (setting-only / action-only captions, track identity). Shared warm-up, three typed branches = the independent offline prototypes above.

**Offline vs. online per type**: event latents offline first (one event clip + short lead-in, causal attention, latents at the end; Streamo standalone captions), then online on full streams (VST delta-style thoughts; teaches non-redundancy with history). Context latents likewise offline first, with longer windows (1-3 min) and scene-level targets (VST establishing thoughts, scene summaries distilled from Streamo captions); the only online-specific part is update dynamics across scene changes, same gap as event latents. Keep causal attention and a bounded per-unit frame buffer in both stages so nothing changes at the transition.

### 4.3 Visual alignment loss

Keep a visual alignment term alongside caption-from-latent throughout, so latents retain detail that captions never verbalize.

- Positives = all frames of the unit as a set.
- Negatives by type: event -> other units of the same video (hard negatives, encode what changed) + other videos; context -> other videos / other location spans only (same-video units are false negatives); object -> track identity matching, not frame alignment.
- Point-wise reconstruction is ill-posed (few latents vs. many frame tokens, no correspondence). Use set-level losses: main term = **coverage / one-sided Chamfer** (each frame token's max cosine sim to any latent in the block, averaged; no pairing, no negatives; same form as Latent-VC's grounding reward) over frame tokens pre-pooled into a few clusters per unit.
- Anti-collapse: event -> contrastive vs. other units; context -> small intra-block diversity penalty (no negatives).
- Alternatives if needed: Hungarian/OT matching, or a training-only cross-attention decoder with point-wise loss on its output. Where a target region exists (object latents with a track mask), LVR-style point-wise MSE on the masked patches is simpler and available.

### 4.4 Training order in the final joint model

Latents first, as a loss schedule.
- Stage 0: VLM emits typed latent tokens (event, context; objects later) at ground-truth close positions, fixed intervals for VST. Train caption-from-latent + contrastive grounding; structural-token loss weight 0. **Bottleneck rule**: caption tokens attend only to latent tokens, never raw frames, or the latents go vestigial.
- Stage 1: ramp in structural-token focal loss; jitter boundary positions +-1-2 s against exposure bias; evaluate free-running timing. Check caption quality at predicted vs. GT boundaries.
- Rationale: silence-dominated token loss would swamp a forming bottleneck; latents are reused by cap. 3 and should be validated alone; stable latents give a prediction-error cue for boundaries.

### 4.5 What transfers from related recipes

**Tyler** (arXiv 2606.16360, text-only math reasoning, preprint) is the closest training recipe. Typed operator tokens in the vocab; on emission, a shared LoRA synthesizer with type-specific query tokens + projection head produces a fixed latent block (8/4/4) appended to context. Stage 1: backbone frozen, operators placed at heuristic boundaries, loss = next-token CE on the *following text only* (latents are context, not targets). Stage 2: GRPO learns when/which/how-many with a budget penalty + an anchoring aux loss for sparse invocations.
- Transfers to us: typed > shared (+3) and swapping types hurts (-5); stage 1 alone gives most of the gain, so test caption-from-latent without a visual term first; expect the sparse-invocation credit problem for close tokens.
- Does not transfer: latents are transient (consumed within one answer), no long-horizon retention or retrieval, no streaming, 1.5-4B models.

**Latent-VC** (sec. 10 for the code): its contrastive cache alignment is the loss we keep; its cache tokens are per-video and query-conditioned, we need per-unit and query-free; it has no retention test, so we add one.

## 5. Caps. 3-5 Design: Graph Construction, Modulation, Retrieval

Idea stage. No graph-structure labels exist in any dataset (sec. 6); the plan is to regenerate VST's KG and evidence paths as pseudo-labels (sec. 9).

**Cap. 3, graph construction**
- **Rerun VST stages 0-1 on our videos to get graph labels.** Their code is complete (sec. 9). Keep `kg_*.json` and `path_entries` instead of discarding them: triples = pseudo-labels for cap. 3, evidence paths = retrieval targets for cap. 5. Swap Gemini for an open VideoLLM to avoid the license clause and cost. Run on Ego4D/EgoTracks so object tracks and KG entities can be cross-linked.
- **Fix entity resolution before trusting the graph.** Add an embedding-based merge over entity names and descriptions after extraction; otherwise node identity drifts on long videos and cap. 3 labels are noisy.
- **EgoTracks identity as node supervision.** Two events sharing a tracked object should share or link a node. Cheap auxiliary loss for cap. 3.
- **Multi-granularity clips.** Run VST stage 1a at e.g. 8 s / 32 s / 128 s clips and keep the three graphs. Coarse-to-fine containment between them gives weak labels for cap. 1 and a hierarchy for cap. 3.
- **Salvage structure from released VST CoT.** The streaming thoughts are entity grounded text. An NER/relation pass recovers a partial graph for free, useful as a sanity check before paying for regeneration.

**Cap. 4, topology modulation**
- **Self-supervision only.** No labels exist. Candidate signal: shorten edges between nodes that land on the same VST evidence path, i.e. edge weight predicts future co-retrieval. Hebbian-style update with QA loss as teacher.

**Cap. 5, query-aware retrieval**
- Retrieval targets: VST `path_entries` (regenerated). Weak proxies: ST-Evidence / Open-o3 evidence spans.
- **Staging.** Evidence-grounded sets (ST-Evidence, Open-o3) fit an RL stage with verifiable rewards after SFT, mirroring both papers' SFT -> RL recipe.
- Streamo's event grounding and time-sensitive QA tasks are query-conditioned and are held back from cap. 1 training for use here.

## 6. Datasets

Candidates for training caps. 1-2. None supervise the graph (caps. 3-5) directly.

| Dataset | Scale | What it gives | Feeds cap. | Notes |
|---------|-------|---------------|-----------|-------|
| Streamo-Instruct-465K | 465.8K samples / 135.9K videos | 1 fps; `<Ns-Ms>` event boundaries; 5 tasks: real-time narration (per-second), event caption, action caption (COIN/YouCook2 steps), event grounding, time-sensitive QA. Sources: Koala, LLaVA-Video, ActivityNet, QVHighlight, YouCook2, HACS, EgoTimeQA, DiDeMo, COIN | 1, 2 | **"3 granularities" = response states `<Silence>/<Standby>/<Response>`, not time scales.** Temporal scale is implicit in task: narration ~1 s, action steps ~10-30 s, events ~30 s-min. Trained with focal loss on state tokens. |
| E.T.-Instruct-164K | 164K samples | Referring, grounding, dense captioning, complex event understanding; paired with E.T. Bench (7.3K, 12 tasks) | 1, 2 | Offline clips, not streaming. Overlaps with Streamo (Streamo re-annotates existing sets, so check for leakage with E.T. Bench eval). |
| EgoTracks | 22K tracks / 5.7K ~6-min Ego4D videos | Long-term single-object tracks with occlusion, exit/re-entry, appearance change | 2 (object nodes) | Gives persistent object identity over minutes: natural supervision for "same node across events" in cap. 3. Egocentric only. |
| VST (Video Streaming Thinking) | 100K streaming-thought + 50K open QA (SFT); 11K (RL). Videos from LLaVA-Vid, Video-Marathon | Per-clip (head, relation, tail) triples from Gemini -> NetworkX KG -> DFS multi-hop paths -> QA + streaming CoT (<=10% entity overlap between chains) | **3, 5** (and 2) | KG triples and evidence paths are not in the HF release, but the full generation code is in `data_gen/` (sec. 9). Release also bundles ~500 GB of source videos (HD-VILA, molmo2, LLaVA-Video-178K, longvila, YouCook2, MovieChat, ActivityNet; Ego4D excluded by license). |
| ST-Evidence-Instruct-160K (Salesforce) | ~161K: 20K CLEVRER + 141K ViCaS | QA + temporal segments + tracked object masklets (SAM 3); Gemini-generated | 2, 5 (spatial evidence) | Joint answer + spatio-temporal evidence. CC-BY-NC; "not for models competing with Google." ViCaS clips are short, so weak for long-horizon memory. |
| Open-o3-Video (STGR-CoT-30k / STGR-RL-36k) | 66K total; sources: VideoEspresso, Video-R1, TVG-R1, TimeR1, TreeVGR, GQA | QA + timestamped keyframes + boxes + CoT linking evidence to answer; verifiable rewards for RL | 2, 5 | Apache 2.0. Short-to-medium clips. Good for RL-with-evidence-reward stage, not for long streams. |
| Ego4D Goal-Step | 48K step segments / 430 h | goal > step > substep tree, step completion, step-to-goal relevance | 1 (goal shifts, levels) | Supplies the hierarchy for level assignment (sec. 3). Egocentric; overlaps EgoTracks and VST source videos. |
| Kinetics-GEBD | | Generic event boundaries | 1 (perceptual shifts; eval) | |

**Object-latent candidates, not egocentric-only** (Vatsal; counts unverified, check before use)
- Scale / open-vocab tracks: SA-V (SAM 2 set; masklets with disappearance and re-entry, no names), TAO + BURST (long-tail categories, long clips).
- Occlusion stress tests: OVIS, MOSE.
- Objects + timed relations (bridge to cap. 3 edges): VidOR, Action Genome.
- Already in pool: ST-Evidence-Instruct masklets (ViCaS), Open-o3-Video boxes.
- EgoTracks / Ego4D stay in as one source, valued for overlap with VST and Goal-Step.
- Note: masklet sets have no names, so object identity is trained visually; names attach later from captions / KG.

**Coverage map**

| Cap. | Direct supervision | Weak / proxy |
|------|--------------------|--------------|
| 1 Segmentation | Streamo (3 granularities), E.T., Goal-Step, Kinetics-GEBD | VST clip boundaries |
| 2 Latent memory | Streamo, E.T., ST-Evidence, Open-o3 | EgoTracks |
| 3 Graph construction | VST KG triples (regenerate with their code) | EgoTracks identity |
| 4 Topology modulation | none | VST streaming CoT (what to remember) |
| 5 Query-aware retrieval | VST `path_entries` (regenerate) | ST-Evidence / Open-o3 evidence spans |

Gaps: cap. 4 has no supervision anywhere. No long-video (>10 min) eval that stresses graph retrieval. Mixed licenses: ST-Evidence is NC with a Gemini clause; VST and Streamo are also Gemini-derived, so check their cards before any release.

## 7. Open Questions

One list, grouped by capability. Resolved items move to sec. 8.

**Cap. 1, segmentation**
- Base model: Qwen3-VL 4B (Streamo's choice) vs. smaller.
- Whether `<standby>` earns its place or is just noise for query-free segmentation.
- Which benchmarks measure segmentation at multiple granularities beyond Streamo held-out sources and Kinetics-GEBD?

**Cap. 2, latent memory**
- Latent capacity: single close-token state vs. a few tokens per unit; matters for cap. 3 retrieval.
- Context latents: static background only, or slowly changing room state too?
- Bounded memory: node budget and eviction/merging policy, and whether that policy is learned.

**Caps. 3-5, graph**
- Node granularity: one node per event? per object? hierarchical (coarse event -> fine sub-events)?
- Edge semantics: temporal adjacency, co-occurrence, learned similarity, or a mix with typed edges?
- How is "edge shortening" (cap. 4) parameterized: scalar weights, metric embedding distance, discrete rewiring?
- How does the query modulate topology (cap. 5): attention over edges, reweighting, or re-running a small GNN conditioned on the query?
- Can we supervise the graph only through downstream QA and still get interpretable topology?
- Which benchmarks cover long-video QA with event structure?

**Supervision and training, cross-cutting**
- Which capabilities get direct supervision vs. emerge from the QA loss alone? Boundary labels exist (e.g. event seg datasets), but graph structure labels do not.
- Curriculum: train capabilities 1-5 jointly end-to-end, or stage them (seg -> memory -> graph -> QA)? Partially answered for caps. 1-2 by sec. 4.4; open for 3-5.
- Online constraint during training: use truncated BPTT over the stream, or train offline on clips and only *deploy* online? Partially answered by the offline-then-online plan in sec. 4.2.

**Data**
- Which video pool do we run VST graph extraction on: VST's bundled videos (cheap, validates the rerun), Streamo videos (aligns with boundary labels), or Ego4D (aligns with EgoTracks)? Decision pending.

## 8. Decisions Log

| Decision | Rationale |
|----------|-----------|
| Started these notes (first as notes.md, now `morph_graphs/vatsal/morph_notes.md`) | Keep a running record of brainstorms |
| Framed problem as 5 capabilities (sec. 1) | From Vatsal's starting notes |
| Candidate dataset pool: Streamo, E.T., EgoTracks, VST, ST-Evidence, Open-o3 | See sec. 6 coverage map |
| ~~Treat VST KG triples as unavailable; rebuild in-house~~ Reversed: use VST `data_gen/` code to regenerate KG + paths | Code is released; only the KG json and `path_entries` are missing from the HF data (sec. 9) |
| Video pool for graph extraction: **open** | See sec. 7 Data |
| Cap. 1 is **query-free**; train as Streamo/VST-style proactive generation with MeCo structural tokens (sec. 3) | Vatsal's call; scale becomes a property of the stream via level-tagged close tokens |
| Boundaries typed by shift kind (spatial / goal / perceptual); hierarchy from goal nesting | Event Segmentation Theory framing; Ego4D Goal-Step supplies the tree |
| The VLM emits latents itself as typed output tokens, no external memory module (sec. 4.1) | Departure from prior typed-memory work; reuses the VST thought-emission mechanism |
| Object latents not egocentric-only; add third-person track datasets | Need many objects leaving and returning; see sec. 6 |
| **Prototyping phase**: train event / context / object latents independently, offline; final model is joint + online (sec. 4.2) | Debuggability; each has its own data and metric |
| Latents are per unit, fixed-size block, with a set-level (coverage / Chamfer) visual alignment term kept throughout (sec. 4.3) | Point-wise reconstruction is ill-posed; captions never verbalize all detail |
| Final joint model trains latents first, structural-token loss ramped in later (sec. 4.4) | Silence-dominated token loss would swamp a forming bottleneck |
| First latent experiment = reproduce Latent-VC stage 1 (LVR fallback), probe released checkpoint first (sec. 2) | Code + weights exist; validates the latent mechanism before typing |

## 9. Reference: VST Data-Gen Pipeline

Code: `data_gen/` in https://github.com/1ranGuan/VST. Release: HF `Catalan258/VST-Training-Data`.

**What the release contains vs. not**
- Has: SFT jsonl (`streamvlm_stage2_75k_vm`, `_17k`, `llava_video_50k`, aux sets), RL `train.jsonl`, bundled videos.
- Missing: `kg_gemini3f_<video>.json` (triples) and QA `path_entries` (evidence chains). `collect_format.py` keeps only question/answer + per-clip thoughts. To get the graph we must rerun stages 0-1.

**Stages** (`scripts/run_stream_thought.sh`)

| Stage | Script | What it does |
|-------|--------|--------------|
| 0 | `tools/split_clips.py` | Equal-length clips: 8 / 16 / 32 for duration <256 s / <1024 s / <=2048 s; longer dropped. Clips are 16-64 s. Audio stripped. |
| 1a | `mhqa/osvGraph.py` `StreamGraphThinker` | Per clip, Gemini 3 Flash sees the **raw clip** (not captions). Context = previous 1 clip + response (`MAX_HISTORY_TURNS=1`) + registry of last 50 entity names. Prompt forces dense scene graph: no hub-and-spoke on the person, mandatory object-object spatial/containment edges, no pronouns. Output `{subject, relation, object, description}`. Stored in `nx.MultiDiGraph`; edge attrs = relation, desc, start/end time, step, clip file. |
| 1b | `mhqa/qaGenerator.py` | Undirected view; skip video if diameter <2. BFS-ish path search (random target depth 2-4, >=4 nodes, Jaccard <0.2 vs. accepted paths), 2 paths/video. For each path edge: (triple, time range, clip video) interleaved into one Gemini call with 2-3 sampled reasoning dimensions (12 types: OCR, spatial, attribute evolution, frequency, procedural, duration, episodic retrieval, consistency, causal, goal, predictive, counterfactual). Output `{question, cot, answer, reasoning_type}` + `path_nodes` + `path_entries`. |
| 2 | `mhqa/stage_3_add_discription.py` | Per-clip streaming thought: delta-only description conditioned on that clip's KG entities and prior thoughts. No question. |
| 3 | `mhqa/collect_format.py` | Chunk thoughts into 128 s windows (duration/8 above 17 min); prior thoughts passed as text `previous`; QA attached to last chunk only. |

**Graph semantics**: nodes = entities (string names), edges = timed events. So it is an entity graph with event edges, not an event graph. Temporal order is recoverable from edge timestamps.

**Access and local copies**
- HF repo is public and ungated; anonymous download works.
- Partial local copy at `morph_graphs/VST-Training-Data/` (Anubhav, 505 MB): all `*_valid` SFT jsonl, RL `train.jsonl`, CLEVRER videos, zip manifests for the rest, 126 sample mp4s under `extracted/`. Anubhav also has a browser at `morph_graphs/anubhav/vst_viewer/`.
- **Our copy**: all annotations (every SFT train+valid split, RL data, zip indexes/checksums, setup scripts; ~1 GB) at `/fs/nexus-scratch/vatsalag/mem_open_vocab/datasets/VST-Training-Data`, symlinked as `morph_graphs/morph_graph/data/VST-Training-Data`. Note the scratch path resolves to `/fs/cfar-projects/mem_open_vocab`, which was 98% full (188 GB free).
- Not local anywhere: the ~500 GB video bundle (hdvila 9 parts, molmo2 2 parts, plus 12 smaller zips). Would not fit on cfar-projects as-is. Ego4D must come from ego4d-data.org.
- VST code is cloned at `morph_graphs/baselines/VST` (with VST-7B weights, eval logs) and `/fs/nexus-scratch/vatsalag/VST`; neither has `data_gen/`, pull to get it.

**What the SFT annotations contain** (inspected `streamvlm_stage2_75k_vm_0124_valid`, 396 videos / 2255 chunks)
- `text_stream`: full streaming thoughts, one per clip `[start, end, text]`; median clip 38 s, median thought ~720 chars; every chunk populated. Delta-style, entity grounded. Usable directly as cap. 2 targets.
- `previous`: all prior thoughts as concatenated text, so each row is self-contained.
- `qa_stream`: on the last chunk only, `[t, t, question, answer]`, ~1.4 QA per video. No CoT and no evidence path; those live in `path_entries` of the stage-1 output, which the formatter drops. That is the only thing regeneration adds.

**Quirks to handle if we reuse the code**
- `save_to_json` appends the cumulative `event_log` every step, so the saved KG json has step-k events duplicated (N-k) times. Dedupe on load, or `load_from_log` will add parallel edges.
- Entity resolution is string-equality via prompt; no embedding merge. Registry truncates to last 50 names, so long videos drift.
- Vertex AI `GenerativeModel` is hard-wired; swapping to an open VideoLLM via vLLM means replacing `load_clip` / `send_message` only (~50 lines).
- Clip granularity (16-64 s) is one level. For cap. 1 we would run at two or three clip sizes.
- Ego4D is in their source list (not bundled), which overlaps with EgoTracks videos.

## 10. Reference: Latent-VC Setup and Codebase

**Setup on cluster** (all verified). Repo clone: `morph_graphs/vatsal/Latent-VC`.
- Models in `model_hub/`: `Qwen3.5-9B-Base` (19.3 GB, 4 shards) and `Latent-VC-9B` (18.8 GB). Repo symlinks `models/Qwen3.5-9B-Base`, `checkpoints/Latent-VC-9B`.
- SFT data: `datasets/Open-o3-Video/videos/stgr/` (58 GB extracted, 40,404 files; all 22,993 paths in `lvc_sft.json` present, zip deleted). Repo symlink `data/videos/stgr`. Note the Latent-VC-Data SFT tars hold videos only, no key frames; Open-o3's zip has both.
- Eval: `datasets/Latent-VC-Data/Eval/Evaluation/{VSIBench (288 videos, all 5,130 items), VideoMME -> datasets/Video-MME/data (900 videos, all 2,700 items)}` plus the six `eval_*.json`. Repo symlink `src/r1-v/Evaluation`. Other benchmarks and all GRPO data skipped.
- `eval_all_benchmarks.py` is absent from the GitHub repo; copied from the HF model repo into the clone root. It reads `eval_<bench>.json` from EVAL_DIR and videos from `EVAL_DIR/<Bench>/`.
- Env not yet built: requirements pin torch 2.6, transformers 5.2.0, flash-attn 2.8.3, causal-conv1d, flash-linear-attention (Qwen3.5 linear attention), CUDA 12.4.

**Codebase, how it actually works** (~5.4K lines, monkey-patches `Qwen3_5ForConditionalGeneration.forward`)
- *Data*: assistant target = `<lvc>` x K key frames, then `<think>` grounded CoT with `<obj><box>at<t>` text, then `<answer>`. Each `<lvc>` expands to `<|lvc_start|>` + N x `<|lvc|>` + `<|lvc_end|>`, N = visual tokens inside that key frame's bbox. 16 frames at 1 fps, 128-768 tokens/frame.
- *Cache tokens in SFT are plain learned embeddings*, no hidden-state feedback; the "recurrent chain" exists only in the legacy `forced_lvc` decode path. Recommended decode `lvc_reasoning` = append start + 8 x `<|lvc|>` + end to the prompt, one prefill, greedy. Inference loop is hand-written, batch 1.
- *Loss* = CE on text (cache positions ignored) + 0.1 x InfoNCE(proj_head(mean hidden over a key frame's cache tokens), mean post-merger embedding of that key frame), T=0.07, negatives = other key frames of the same video. Proj head = 2-layer MLP at 10x LR. Vision tower + merger frozen, LLM full fine-tune, ZeRO-3, 1200 steps, eff. batch 32, lr 1e-5, sdpa not flash.
- *Caveat*: loss groups cache hidden states by whole-frame token counts but the inserted count is the bbox token count, so per-key-frame alignment is exact only for full-frame boxes; check with a dry run before trusting `loss_lvc`.
- *GRPO*: no contrastive term; rewards = accuracy, format, temporal grounding (timestamps vs key frame times), latent reward (cos sim of cache hidden mean vs key frame mean, thresholded). Custom 1.3K-line trainer. We skip it.
- *Hooks for us*: typed latents = new special tokens + `replace_lvc_tokens`; per-unit alignment = replace per-key-frame pooling in `forward_sft.py`; boundary emission = new decode loop (the existing one is ~100 lines). Cache hidden states are reachable via `output_hidden_states` for the retention probe.

## 11. References

Datasets and benchmarks
- Streamo: Streaming Video Instruction Tuning (CVPR 2026). https://arxiv.org/abs/2512.21334
- E.T. Bench / E.T.-Instruct-164K / E.T. Chat (NeurIPS 2024). https://arxiv.org/abs/2409.18111
- EgoTracks (NeurIPS 2023 D&B). https://arxiv.org/abs/2301.03213
- Ego4D Goal-Step (NeurIPS 2023 D&B): 48K step segments / 430 h, goal > step > substep, plus step completion and step-to-goal relevance. https://proceedings.nips.cc/paper_files/paper/2023/file/7a65606fa1a6849450550325832036e5-Paper-Datasets_and_Benchmarks.pdf
- VST: Video Streaming Thinking (2026). https://arxiv.org/abs/2603.12262 ; code https://github.com/1ranGuan/VST (pipeline in `data_gen/`) ; data https://huggingface.co/datasets/Catalan258/VST-Training-Data
- Evidence-Backed Video QA / ST-Evidence-Instruct (Salesforce, 2026). https://arxiv.org/abs/2607.11862 ; https://huggingface.co/datasets/Salesforce/ST-Evidence-Instruct
- Open-o3-Video (ICML 2026). https://arxiv.org/abs/2510.20579 ; https://huggingface.co/datasets/marinero4972/Open-o3-Video
- Kinetics-GEBD (generic event boundary detection) for cap. 1 eval. https://arxiv.org/abs/2101.10511

Methods
- MeCo: Measure Twice, Cut Once (ICLR 2026). https://arxiv.org/abs/2503.09027
- Tyler: Typed Latent Reasoning for Language Models (2026 preprint; typed latent operators invoked mid-decoding, 2-stage frozen-backbone LoRA + GRPO). **No code, weights, or data released** (project page links only the paper). https://arxiv.org/abs/2606.16360
- LVR (ICLR 2026; fixed budget of 4/8/16 latent steps, SFT = point-wise MSE to patch embeddings inside an annotated ROI box, RL = format + accuracy reward only). https://arxiv.org/abs/2509.24251 ; code https://github.com/Penation/lvr
- CoLVR (2026). https://arxiv.org/abs/2605.08802
- Latent Visual Cache for Video Reasoning / Latent-VC (2026, Qwen3.5-9B; 8 recurrent cache tokens emitted once per video after the prompt, blocks contrastively aligned to annotated key-frame patch embeddings, then GRPO with a grounding reward; per-video and query-conditioned, not per-frame). https://arxiv.org/abs/2607.02607 ; code https://github.com/BRZ911/Latent-VC
- LatentStream (2026, streaming hierarchical latent memory). https://arxiv.org/abs/2609.04131
- H-Net: Dynamic Chunking (Hwang, Wang, Gu 2025), fallback chunker. https://arxiv.org/abs/2507.07955
