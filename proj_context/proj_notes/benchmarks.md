# Benchmarks — Candidate Eval Sets

Shortlist for evaluating **query-adaptive event-graph memory** on multi-hop / long-video QA. Rubric per benchmark:

1. **# QA pairs** (too small → noisy) · 2. **Video duration** · 3. **Venue** · 4. **Example question types**

The extra column we care about — **evidence/event localization** (does it annotate *where* in the video the answer's evidence lives?) — since that aligns directly with graph traversal / retrieval grounding. CG-Bench is the exemplar here.

## Comparison Table

| Benchmark | # QA | Duration | Venue | Evidence localization | Domain |
| --- | --- | --- | --- | --- | --- |
| **CG-Bench** | 12,129 | 10–80 min (long) | **ICLR 2025** | ✅ clue **interval** per QA | open-domain (web) |
| **CaST-Bench** | 2,066 (1,015 vids) | short–medium, open-domain | **CVPR 2026** | ✅ **temporal segments + bbox tracks** (causal chain) | open-domain |
| **EgoMemReason** | 500 | week-long (~25.9 h backtrack) | COLM 2026 | ✗ **answers not released** (private key, leaderboard-only) **and** ✗ no evidence spans (only `query_time`) | egocentric (EgoLife) |
| **SuperMemory-VQA** | 4,853 | 52.9 h total, >1 h context | arXiv 2026 | ✅ **answer evidence time spans** released (+ answer key); 34% multi-evidence | egocentric (Aria) |
| **EGOSTREAM** | 2,250 → 8,528 evals | up to 45.3 h (AVW) | arXiv 2026 | ◐ Answer Validity Window (not clue spans) | egocentric (multi-source) |
| **EgoLifeQA** | 3,000 | 44.3 h/clip (week, 6 people) | **CVPR 2025** | ◐ evidence timestamps | egocentric (EgoLife) |
| **EgoLongQA** | 700 val (+ hidden test) | **10+ min** egocentric clips | **ECCV 2026** Wearable AI Workshop | ✗ (task requires temporal recall; **no** clue intervals in released schema). ✅ answers released on val | egocentric (Meta Wearable AI) |
| **Video-Holmes** | 1,837 | 1–5 min (short) | arXiv 2025 (venue unconfirmed) | ✗ (clues implicit, not localized) | suspense short films |
| **VideoEspresso** | ~200 K train / smaller test, 14 tasks | short–medium | **CVPR 2025 (Oral)** | ◐ core-frame + spatial/temporal CoT | open-domain |

✅ = explicit, per-QA localized evidence · ◐ = partial / coarse · ✗ = none

## Per-benchmark notes

### CG-Bench — *top pick*
- **12,129** human-annotated **QA-clue triplets** over **1,219** videos, **10–80 min**. Largest long-video QA bench.
- **ICLR 2025**. arXiv 2412.12075.
- **Clue-grounded**: every QA carries the **time interval(s)** where the answer's evidence lives → white-box (feed clue) + black-box (must retrieve) eval. Directly measures whether retrieval lands on the right moments.
- Question types: **perception, reasoning, hallucination** (perception split has 10 subtypes).
- *Why it fits:* the clue intervals are exactly the "where did the event occur" supervision you liked — ideal for scoring graph traversal / node retrieval, not just final answer.

### CaST-Bench — *strong second (grounded multi-hop)*
- **2,066** questions over **1,015** videos; causal chains annotated with **temporal segments + bounding-box tracks**.
- **CVPR 2026**. arXiv 2605.23216.
- Task = pick answer **and** localize the **chain** of spatio-temporal evidence (trigger → effects). Metric includes a **faithfulness/grounding** score, not just accuracy.
- *Why it fits:* multi-hop causal chains with explicit evidence grounding = natural test for edge-traversal over an event graph. Videos are shorter/open-domain, so less about ultra-long memory, more about relational grounding.

### EgoMemReason — *memory-reasoning fit, but small N + no released grounding*
- **500** MCQs on week-long EgoLife video; **25.9 h** backtracking. **COLM 2026** (arXiv 2605.09874).
- Three memory types → six challenges: **entity** (cumulative state tracking, temporal counting), **event** (ordering, linking), **behavior** (spatial preference, activity pattern).
- **Answers: not released.** Public file (`annotations_public.jsonl`) is questions + options only; the correct answers live in a **private key** and you can *only* obtain a score by submitting to the leaderboard Space. No local/offline evaluation, no per-question correctness.
- **Evidence localization: not released either** — schema is `question / options / query_type / query_time`. The "5.1 evidence segments/Q" is a *construction statistic*, not a per-item annotation; you get *when the question is asked* (`query_time`), not *where the answer lives*.
- *Why it fits:* "event linking / ordering" is precisely inter-event relational reasoning. Caveats: **500 Q is small** → per-split (50–100/split) noisy; **can't** score retrieval/traversal grounding; and the **held-out-answer / leaderboard-only** setup means no offline iteration — a real friction for development.

### SuperMemory-VQA — *good size + long-horizon + released evidence spans*
- **4,853** human-verified QA over **52.9 h** (10 people, Meta Aria glasses); **34%** multi-evidence. arXiv 2026 (2606.00825).
- Six tasks: object/location memory, intent recall, visual scene recall, timeline reconstruction, conversational memory, in-context retrieval. **Ordered MCQ** (correct > vague > wrong > **unanswerable**) → tests abstention/hallucination. Metrics: Ans-F1, QA-Acc, QA-MRR.
- **Releases evidence localization**: each item ships the correct answer, `question timestamps`, **`answer evidence time spans`**, and evidence modalities → you *can* score whether retrieval/traversal lands on the right moments (unlike EgoMemReason). Answers are public (zero-shot setting).
- *Why it fits:* decent size + genuinely long-horizon + **released per-QA evidence spans** + multi-evidence. Along with CG-Bench, one of the few that gives the grounding supervision you want; multimodal (audio/gaze/SLAM) is a bonus, out of current scope.

### EGOSTREAM — *streaming diagnostic*
- **2,250** questions expanded to **8,528** recall-conditioned evals; **7** dims (detail, spatial, temporal, event, social, causal, prospective); AVW up to **45.3 h**.
- arXiv 2026 (2605.31557); sources Ego4D/EgoLife/EgoTempo/Multi-Hop EgoQA/HD-EPIC.
- **Query-agnostic streaming** (answer using only content seen so far) + **Answer Validity Window** separates forgetting from world-state change. Explicitly a **memory-management testbed** (pruning/merging/offloading).
- *Why it fits:* great for the *streaming* framing and for diagnosing memory decay, but it's built to compare KV-mechanisms, not evidence retrieval — evidence isn't clue-localized the way CG-Bench is.

### EgoLifeQA — *ultra-long baseline*
- **3,000** QA (500 × 6 participants), **44.3 h/clip**, week-long; 66% need >2 h look-back, 15% >24 h.
- **CVPR 2025**. arXiv 2503.03803. Five types: EntityLog, EventRecall, HabitInsight, RelationMap, TaskMaster.
- *Why it fits:* canonical ultra-long egocentric memory bench with evidence timestamps; pairs naturally with EgoMemReason (same source video).

### EgoLongQA — *wearable long-video MCQ (answers released; no clue spans)*
- **700** val MCQs (equal-size hidden test for the challenge) over **10+ min** egocentric clips. Part of Meta's **Wearable AI** dataset / **ECCV 2026** Wearable AI Workshop Grand Challenge (`facebook/wearable-ai`, config `egolongqa`). ~203 GB of video for this task.
- Schema releases **`answer`** (open-ended) + **`mcq_answer`** (A–D) + activity `category` — so **offline scoring on val is possible** (unlike EgoMemReason).
- **No released evidence intervals** — fields are `video_path / question / answer / mcq_options / mcq_answer / category`. Workshop framing says the task needs "temporal grounding and recall of events spanning the full video," but clue spans aren't in the public annotation.
- Example type: aggregate / recall over the clip (e.g. "How many sets of workouts have I done so far?" → "Ten sets — three each of push-ups, bicep curls, and lateral raises, plus one set of front raises.").
- Sibling tasks in the same release: **EgoConv** (multi-turn, has `video_intervals`) and **EgoProactive** (streaming interruption decisions).
- *Why it fits:* egocentric long clips + released answers. Caveats: **700 is small** (noisy); duration is "10+" not hour-scale; **can't score retrieval grounding** without clue intervals.

### Video-Holmes — *conceptually aligned, but short + ungrounded*
- **1,837** Q over **270** suspense films, **1–5 min**. Requires actively locating + linking **multiple scattered clues** (7 tasks: social reasoning, intention/motive chaining, timeline analysis, multimodal hint, physical anomaly, theme, etc.).
- arXiv 2505.21374 (venue not confirmed).
- *Why it fits (weakly):* multi-clue linking is on-theme, but videos are short (fits in context → little memory pressure) and clues aren't time-localized, so it can't score retrieval grounding.

### VideoEspresso — *mostly a training CoT dataset*
- Large (~200 K QA) with multimodal CoT + **core-frame selection** and spatial/temporal grounding; **14** reasoning tasks. Test benchmark is a smaller curated subset.
- **CVPR 2025 (Oral)**. arXiv 2411.14794.
- *Why it fits (weakly):* useful as **training/CoT supervision** and its core-frame selection echoes node retrieval, but videos are short–medium and it's not a long-video memory bench.

## Additional — evidence-grounded, since Mar 2026, ≥20 min avg

Filtered to the hard constraints (release ≥ Mar 2026 **and** avg duration ≥20 min) with per-QA evidence grounding. These are the strongest fits for the retrieval/traversal claim on long video.

| Benchmark | # QA | Avg duration | Released | Evidence localization | Task |
| --- | --- | --- | --- | --- | --- |
| **LongVidSearch** | 3,000 | **~26 min** | arXiv 2603 (Mar 2026) | ✅ **necessary** evidence clips, Hop-2/3/4 | agentic multi-hop retrieval planning |
| **CoMET-Bench** | 2,789 (600 vids) | **33.8 min** | arXiv 2606 (Jun 2026) | ✅ timestamps of **all** qualifying events | conditional multi-event grounding |
| **ExtremeWhenBench** | 2,273 (194 vids) | **75.7 min** (max 9 h) | arXiv 2606 (Jun 2026) | ✅ interval per NL query | hour-scale temporal grounding |

### LongVidSearch — *best new fit*
- **3,000** QA over **447** videos, **~26 min** avg. arXiv 2603.14468 (Mar 2026).
- **Retrieval necessity**: Hop-*k* questions need exactly *k* necessary evidence clips — removing any one makes the question unsolvable. Four categories: State Mutation, Causal Inference, Global Summary, Visual Tracking.
- Standardized **tool interface** (`Search_Clips_In_Video` / `Get_Clip_Detail` / `FINAL_ANSWER`); reports accuracy **and** tool-call cost. GPT-5 tops out at 42.4%; with gold clips it's near-perfect → **retrieval planning is the bottleneck**.
- *Why it fits:* this is almost exactly our thesis — multi-hop evidence retrieval over long video where similarity-only/monolithic approaches fail. Directly scores whether traversal lands on the right (necessary) clips.

### CoMET-Bench — *multi-event grounding*
- **2,789** queries over **600** videos, **33.8 min** avg, 5 domains (Sports, TV/Movie, Life Record, Knowledge, Surveillance). arXiv 2606.15320 (Jun 2026).
- Each query composed from 4 temporal + 3 spatial conditions; annotations ground **every** qualifying event instance (not a single moment). Includes a **negative-query** subset + Rejection-F1 to catch "always-empty" gaming.
- *Why it fits:* exhaustive multi-event retrieval with compositional conditions — tests aggregation across scattered evidence, close to multi-hop traversal.

### ExtremeWhenBench — *hour-scale grounding as search*
- **2,273** open-form queries over **194** videos, **mean 75.7 min** (max 9 h). arXiv 2606.12300 (Jun 2026).
- Returns interval \([t_s,t_e]\) per query. Key result: **85% of failures are search failures**, and a retrieve-then-ground hybrid beats monolithic Video-LLMs **6.7×** — mirroring retrieve-then-read in open-domain QA.
- *Why it fits:* empirical backing for the "memory = search/retrieval, not raw context" framing; strong motivation citation even if it's grounding-only (no QA answer).

**Excluded by the filter:** EG-VQA (arXiv 2606, but avg **177.6 s ≈ 3 min** → fails duration); HLV-1K (hour-long, but arXiv 2501 / **Jan 2026** → fails date); CaST-Bench (May 2026, but open-domain short clips → fails duration).

## Recommendation

- **Primary:** **CG-Bench** — large, accepted, long videos, and per-QA **clue-interval** grounding lets us score *both* answer accuracy and retrieval/traversal correctness.
- **Best new match (≥20 min, 2026):** **LongVidSearch** — retrieval-necessary multi-hop over ~26 min videos is our thesis in benchmark form; **CoMET-Bench** (33.8 min, multi-event) and **ExtremeWhenBench** (75.7 min, search framing) reinforce it.
- **Grounded multi-hop:** **CaST-Bench** — causal chains with temporal + spatial evidence; best for testing edge-traversal, accepting shorter videos.
- **Ultra-long memory (grounded):** **SuperMemory-VQA** — long-horizon egocentric *with released answer-evidence time spans* + public answer key, so it doubles as a grounding eval. **EgoMemReason** only as an event-linking diagnostic (small N, **answers held out / leaderboard-only**, no released grounding); **EgoLifeQA** as the established baseline on the same EgoLife video.
- **Streaming stress-test:** **EGOSTREAM** if/when we frame the method as online/streaming.
- **De-prioritize** Video-Holmes (short, ungrounded) and VideoEspresso (training set / short) for the memory-retrieval claim; keep VideoEspresso in mind for CoT training data.
