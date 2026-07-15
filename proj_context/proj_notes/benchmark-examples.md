# Benchmark Examples — Question Style by Category

Real examples sampled from local annotation files. One per category — for taste, not coverage. Each example has an **explicit ID** you can grep in the source file.

**Sources & ID keys:**
| Bench | Path | Lookup key |
| --- | --- | --- |
| EgoMemReason | `/fs/nexus-scratch/vatsalag/mem_open_vocab/datasets/EgoMemReason/annotations_public.jsonl` | `example_id` (also `p_id`) |
| CG-Bench | `/fs/nexus-scratch/vatsalag/mem_open_vocab/ReKV/data/cgbench/full_mc.json` | `video_id` + `conv_idx` (0-based index into `conversations`) |
| EgoLongQA | `morph_graph/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl` | `video_path` (also 0-based `line`) |
| SuperMemory-VQA | `super_memory_vqa_qa.json` | `question_id` |

---

## 1. EgoMemReason (500 MCQs; answers held out)

Style overall: **first-person, house-coliving memory**, asked at a `query_time`, looking *back*. Options 4–10. Needs multi-day evidence aggregation.

### Entity — Cumulative State Tracking (100)
**ID:** `example_id=51` · `p_id=A1_JAKE_DAY2_21_26_55_q051`
> Looking at this wine in front of me now, where was it when we had a meeting in the main room yesterday noon?
> A. projection screen · B. kitchen sink · C. dining table · D. sofa · E. bedroom door · F. whiteboard
>
> `query_time`: DAY2, 21:26:55

### Entity — Temporal Counting (100)
**ID:** `example_id=300` · `p_id=A1_JAKE_DAY2_12_50_00_q300`
> How many chairs have you seen in the main room in the 1st floor up to now?
> A. 8 · B. 6 · C. 9 · D. 7 · E. 5 · F. 10
>
> `query_time`: DAY2, 12:50:00

### Event — Event Ordering (100)
**ID:** `example_id=401` · `p_id=A1_JAKE_DAY7_23_00_00_q401`
> Order the following events from earliest to latest:
> 1) Have Mincheng Bibimbap for lunch for the first time
> 2) Setup recording for group dancing
> 3) Second time visit Hema Fresh
> … (8 events) …
> Options are permutations like `2 -> 1 -> 3 -> 6 -> 4 -> 5 -> 7 -> 8`
>
> `query_time`: DAY7, 23:00:00

### Event — Event Linking (100)
**ID:** `example_id=150` · `p_id=A3_TASHA_DAY4_13_02_06_q150`
> What was the last event happened on this table?
> A. we made cakes · B. somebody did makeup for me · C. we had dinner · D. we put together wooden pieces · E. we played video games
>
> `query_time`: DAY4, 13:02:06

### Behavior — Spatial Preference (50)
**ID:** `example_id=250` · `p_id=A1_JAKE_DAY7_20_00_00_q250`
> When I do craft building, where does it usually happen?
> A. backyard table · B. orange table · C. gingham table · D. kitchen table · E. bedroom · F. outside

### Behavior — Activity Pattern (50)
**ID:** `example_id=1` · `p_id=A1_JAKE_DAY7_19_00_00_q001`
> What do I most often eat for breakfast?
> A. Pancake · B. Rice · C. Burger · D. Dumplings

**Takeaway:** closest to our multi-hop / inter-event claim (esp. Ordering + Linking), but no released evidence spans and no public answers.

---

## 2. CG-Bench (12,129 QAs; clue intervals released)

Style overall: third-person **"In the video, …"** long-form online video. Every QA has `temporal_windows` (clue intervals). Local file tags are the Level-2 buckets below.

### Entity Perception (2,666)
**ID:** `video_id=BV1s94y1G7RD` · `conv_idx=0`
> In the video, a woman wearing glasses and a brown top folded the lower right corner of the book. What vegetables were on the book?
> Answer: corn · clue: `[5, 7]`s · domain: Life Record

### Event Perception (2,382)
**ID:** `video_id=3ZpmN3-2-Zo` · `conv_idx=9`
> In the video, what are the three little girls in black clothes doing while they are talking?
> Answer: Three little girls opening gifts · clue: `[455, 460]`s

### Event Cognition (1,480)
**ID:** `video_id=BV1s94y1G7RD` · `conv_idx=1`
> In the video, the man wearing glasses, a red raincoat and blue pants is using a ladder for what?
> Answer: To remove the soaked and swollen leaves and fruit from the roof that are blocking the drain · clue: `[60, 82]`s

### Entity Cognition (1,434)
**ID:** `video_id=BV1JH4y1w7Qz` · `conv_idx=4`
> In the video, where did the son take the hammer from?
> Answer: Behind the pillow. · clue: `~[128, 168]`s

### Text Perception (1,238)
**ID:** `video_id=BV1JH4y1w7Qz` · `conv_idx=9`
> What does the couplet that appears when the family of three takes their bow in the video say?
> Answer: Good luck · clue: `~[504, 575]`s

### Time Perception (734)
**ID:** `video_id=3ZpmN3-2-Zo` · `conv_idx=13`
> In the video, at the 584th second, what is the woman playing on the table?
> Answer: Playing with trains · clue: `[1452, 1453]`s

### 2D Spatial Perception (638)
**ID:** `video_id=BV1KK41157zV` · `conv_idx=1`
> In the video, what is that thing on the right side of the game desktop?
> Answer: Three piles of blocks of different colors · multi-window clues

### Scene Perception (615)
**ID:** `video_id=BV1Rm41117YT` · `conv_idx=2`
> …when the protagonist finished shopping and returned to the marble table outside his yard to taste the food, what was the weather like?
> Answer: sunny · clue: `[359, 365]`s

### Hallucination (444)
**ID:** `video_id=BV1KK41157zV` · `conv_idx=13`
> In the video, if the protagonist wants to build a road, what hand does he need? 1. Brick 2. Iron 3. Wooden 4. Gold
> Answer: 1 and 3 · multi-window

### Time Cognition (256)
**ID:** `video_id=BV1KK41157zV` · `conv_idx=5`
> …based on the experience of the protagonist … how long would a board game played by three people normally last?
> Answer: About forty to fifty minutes. · clues: `[100,110]`, `[2160,2650]`

### Text Cognition (192)
**ID:** `video_id=zqQM66uAig0` · `conv_idx=4`
> …when the protagonist says "to get rid of this section here…", where is begin relative to end?
> Answer: above

### Scene Cognition (50)
**ID:** `video_id=BV1Hd4y1T7wK` · `conv_idx=6`
> In the video, where did the protagonist go first and then?
> Answer: Lakeside-Yacht-Shore · many windows over a long drive

**Takeaway:** best **grounding eval** substrate (clue intervals everywhere). Style is perception/cognitive fact QA over long *third-person* video — less “memory assistant” / multi-day life journaling than the ego ones. Cognition + multi-window Time Cognition are the multi-hop-ish subset.

---

## 3. EgoLongQA (700 val; answers released)

Style overall: first-person wearables, often **"After I …, what …"** chaining within a single 10+ min egocentric clip. Schema: `question / answer / mcq_options / mcq_answer / category`.

Note: `category` is a **video activity/domain tag**, not a question-type taxonomy — ignore it for reasoning style.

### Style examples (activity tags ignored)

**Within-clip temporal chaining**
**ID:** `video_path=00cdb8ca10c069f8.mp4` · `line=0`
> After I left the grassy trail that ran alongside the flooded marsh, what was the first man-made structure I encountered on the paved section of my walk?
> Answer: A concrete bench on the left side of the path.

**Conjoined first/later attribute recall**
**ID:** `video_path=0391ffc9c8c38c67.mp4` · `line=9`
> What was the name on the first sign I saw after stepping onto the sidewalk, and what speed limit was indicated on the last sign I saw before stepping off the sidewalk?
> Answer: Jesse Bolling Hall; 25

**ID:** `video_path=0207cc5be310f3ba.mp4` · `line=3`
> What scent was the first candle I picked up, and what shape were the discounted candles I examined later in the seasonal section?
> Answer: Cashmere Vanilla; pumpkin-shaped.

**Object continuity across scenes**
**ID:** `video_path=08910129d9d6ff34.mp4` · `line=19`
> What container did I fill with water in the kitchen that I later carried through the hallway?
> Answer: The black watering can.

**ID:** `video_path=11c6e381488dcff7.mp4` · `line=39`
> After I removed the plastic bag from the closet, where did the purple package that was inside it end up?
> Answer: In the red suitcase.

**Takeaway:** short-horizon **within-clip temporal chaining / attribute recall**. Good vibe check for egocentric “what did I just see,” weak for week-long multi-hop or released grounding.

---

## 4. SuperMemory-VQA (4,853; answer + evidence spans released)

Style overall: natural **"assistant memory"** phrasing (“I’m trying to…”, “Hey Aria…”). Ordered choices: correct / vague / wrong / **unanswerable**. `metadata.skill` + `answer_evidence.time_span`(s).

### object_location_memory (933)
**ID:** `question_id=4`
> I need to take my medicine before I start cooking. Where did I leave that small white bottle?
> Correct: You left it in the kitchen drawer next to the stove.
> Vague: …one of the drawers in the kitchen area.
> (+ wrong, + unanswerable)

### conversational_memory (932)
**ID:** `question_id=1`
> I'm thinking about cooking meat. What did B say he uses his Instant Pot to cook?
> Correct: He said he uses it to cook beef.
> Vague: …to cook meat.

### visual_recall (850)
**ID:** `question_id=7`
> I was just thinking about our Monopoly game the other day. How much money did I get from that beauty contest card?
> Correct: You collected $10 for winning second prize.
> Vague: You collected some cash for winning a prize.

### timeline_reconstruction (750)
**ID:** `question_id=15`
> I'm writing down the recipe steps for a friend. How exactly did I prep the red onion earlier?
> Correct: First … ends off, then peeled … cut in half, finally sliced into thin strips.
> (wrong option often permutes the step order)

### in_context_retrieval (731)
**ID:** `question_id=5`
> Hey Aria, thinking back to our Monopoly game, what was the difference between that $200 rent I paid B and the rent he paid me for St. Charles Place?
> Correct: You paid B $186 more than he paid you for St. Charles Place.
> (needs linking two earlier facts → compute)

### intent_recall (657)
**ID:** `question_id=10`
> I'm peeling these leftover garlic cloves. What did I say I wanted to do with them earlier?
> Correct: You intended to put them into the dish so they would not be wasted.
> Vague: You intended to use the leftover cloves for the meal you were making.

**Takeaway:** strongest **memory-assistant** feel + released evidence spans. Multi-evidence / timeline / in-context retrieval are closest to our traversal story; object-location & conversational are more single-hop lookup.

---

## Quick style comparison

| Bench | Persona | Temporal scope | Multi-hop / relational? | Evidence spans? |
| --- | --- | --- | --- | --- |
| EgoMemReason | egocentric house-mate | days–week | **yes** (order/link/pattern) | ✗ |
| CG-Bench | third-person “in the video” | 10–80 min clip | sometimes (cognition, multi-window) | ✅ |
| EgoLongQA | egocentric “after I…” | within ~10+ min clip | weak (mostly chained recall) | ✗ |
| SuperMemory-VQA | AR memory assistant | hours–days | **yes** (timeline / in-context) | ✅ |
| **morph_vsi (ours)** | object×room trajectory | ~10 min indoor walk | **yes** (sequence / order / nth) | ✅ frame clues |

---

## 5. morph_vsi / VSI-Morph (ours) — 248 MCQs

**Source:** `morph_graph/data/vsi_morph/vsi_events_mc_poc.json`  
**Lookup key:** `qid`  
**Adapter:** `morph_graph/inference/benchmarks/vsi_morph.py`

POC built from VSI-style indoor walkthroughs annotated as **scene (room) + object appearance** events. **25 video configs** = `(video_uid, object)` pairs over **6** underlying clips (`0000`–`0005`) × up to 5 objects (Cat / Golden Retriever / Panda / Parrot / Hello Kitty). Avg ~2.2 `required_clues` per Q; every item has `clue_frames` (frame indices). Answers released (`correct_answer` letter).

Style overall: **object-centric spatiotemporal** — count, locate-nth, contrast rooms across appearances, order rooms between object sightings. Feels closer to structured event-graph QA than open-domain video chat.

### sequence (25)
**ID:** `qid=1` · `video_uid=0002` · object=Golden Retriever
> What rooms were visited in sequence between the fifth and sixth appearances of the Golden Retriever?
> A. Kitchen → Electrical Office Space · … · clue frames include 268/273/315/320 (+ earlier appearances)

### nth_appearance (25)
**ID:** `qid=2` · `video_uid=0002`
> In which room was the Golden Retriever located during its seventh appearance in the video?
> C. Electrical Office Space

### contrast (25)
**ID:** `qid=3` · `video_uid=0002`
> How does the location context of the Golden Retriever differ between its very first appearance and its final appearance?
> A. first = Lab, final = Kitchen · (10 clue frames spanning first→last)

### post_final (25)
**ID:** `qid=4` · `video_uid=0002`
> Which rooms were entered immediately after the final appearance of the Golden Retriever?
> B. Living Room

### object_total_count (26)
**ID:** `qid=70` · `video_uid=0000` · object=Panda
> How many distinct encounters with the Panda occur throughout the video?
> B. 13 times

### object_room_count (25)
**ID:** `qid=105` · `video_uid=0000` · object=Cat
> How many times was the Cat seen in the Office Space?
> A. 7 times

### room_order (25)
**ID:** `qid=102` · `video_uid=0000`
> What was the eighth room visited in the video?
> C. The Industrial Room.

### room_type_count (25)
**ID:** `qid=103` · `video_uid=0000`
> How many bedroom visits occurred in the video?
> C. 2 visits.

### room_visit_order (25)
**ID:** `qid=202` · `video_uid=0000`
> In what order were the Office, the Bedroom/Toy Room, and the Kitchen visited in the video?
> D. Bedroom/Toy Room → Office → Kitchen

### object_seen_order (22)
**ID:** `qid=203` · `video_uid=0000`
> In what order was the Cat first seen across the Office, the Bedroom/Toy Room, and the Industrial Room?
> D. Bedroom/Toy Room → Office → Industrial Room

**Takeaway for Morph Graphs:** this is the **in-house probe** most aligned with event-graph traversal — nodes ≈ (object sighting / room visit), edges ≈ sequence; released `clue_frames` + `required_clues` let you score retrieval grounding. Caveats: **small N (248 Q / 25 video×object configs / 6 clips)**, ~10 min indoor only, object/room vocabulary is synthetic-toyish vs. SuperMemory/CG-Bench scale.