# Token-grid aspect distortion in `smart_resize`

**Status:** hypothesis with supporting evidence, **not** an established result. Generated post-hoc
after observing an accuracy ordering, so it carries no evidential weight on its own — see
[Caveats](#caveats). Recorded because it is mechanistic, computable without reference to any model
output, and makes falsifiable predictions that have since held three times.

## What it is

Qwen3-VL sees a frame as a grid of 32x32-pixel patches (`patch_size 16` x `merge_size 2`).
`smart_resize` builds that grid by picking **one** scale factor to hit the token budget, then
rounding height and width down to whole patches **independently**. The two axes lose different
amounts, so the grid is not the same shape as the source frame and the image is delivered
anisotropically stretched.

1280x720 at `max_tokens=512`:

```
target area = 512 tokens x 1024 px = 524,288 px
scale beta  = sqrt(921600 / 524288) = 1.326

height:  720 / 1.326 = 543.1 px -> floor to 512 (16 patches)   lost 31.1 px = 5.7%
width : 1280 / 1.326 = 965.5 px -> floor to 960 (30 patches)   lost  5.5 px = 0.6%

grid aspect 30/16 = 1.875  vs true 1280/720 = 1.778  ->  +5.5% horizontal stretch
```

The model sees a 16:8.5 world where the video is 16:9. Every object is slightly too wide.

## Metric

```python
rh, rw = smart_resize(h, w, factor=32,
                      min_pixels=min_tokens*32*32, max_pixels=max_tokens*32*32)
gh, gw = rh // 32, rw // 32
distortion = abs((gw / gh) / (w / h) - 1) * 100     # percent
```

Population figures are the **mean of the absolute value** over every video in a dataset. The sign
is informative per-video and is discarded here: 852x480 at 512 is **+2.11%** (too wide) while at
536 it is **-0.58%** (slightly too tall).

This is a constructed metric, not something the model or the paper defines. `|log(ratio)|` or a
per-axis scale error would rank budgets slightly differently; that has not been tested.

## Why it is erratic rather than monotone in the budget

Whether an axis lands just above or just below a patch boundary is arbitrary. Some budgets land
almost exactly on the true aspect; neighbouring ones miss badly. Measured over every video
(103 LVBench, 1219 CG-Bench):

| max_tokens | 720p grid | grid aspect | LVB tpf | LVB distortion | CGB tpf | CGB distortion |
|---|---|---|---|---|---|---|
| 64  | 8 x 15  | 1.875 | 60  | 6.22% | 60  | 6.34% |
| 128 | 8 x 15  | 1.875 | 120 | 5.39% | 120 | 5.49% |
| 256 | 12 x 21 | 1.750 | 247 | 1.76% | 251 | 1.61% |
| 512 | 16 x 30 | 1.875 | 478 | **4.88%** | 474 | **4.02%** |
| 536 | 17 x 30 | 1.765 | 510 | **0.93%** | 510 | **0.71%** |
| 608 | 18 x 32 | **1.778** | 577 | **0.18%** | 576 | **0.12%** |
| 640 | 18 x 33 | 1.833 | 599 | 2.91% | 608 | 2.69% |
| 768 | 22 x 40 | 1.818 | 721 | 1.30% | 720 | 1.34% |

`608` is near-zero because 18:32 **is** 9:16 exactly. `512` is the worst of the high-token budgets.
Nothing about "more tokens" predicts which you get, which is why a tokens-per-frame sweep that
varies only `max_tokens` confounds token count with grid geometry.

Note `min_tokens` is **inert** for any video that downscales (~90% of both datasets): the
`h_bar*w_bar > max_pixels` branch fires first and `min_pixels` is never read. Every "tpf sweep"
run so far has therefore been a `max_tokens` sweep.

## Evidence

Three independent measures, all ordering with distortion rather than token count.

**1. LVBench accuracy, 5 arms, paired on all 1549 questions**

| tpf | distortion | accuracy |
|---|---|---|
| 120 | 5.39% | 47.97 |
| 252 | 1.76% | **49.26** |
| 478 | 4.88% | 47.84 |
| 510 | 0.93% | **49.19** |
| 594 | 2.91% | **49.26** |

The three low-distortion arms land at 49.19-49.26 and the two high-distortion arms at 47.84-47.97,
with no overlap — and the split cuts *across* the token axis (120 and 478 differ 4x in tokens and
score identically). The 510 arm was run as a prediction: it carries 7% more tokens than the 478 arm
and was predicted to score with the low-distortion group. It did.

**2. CG-Bench retrieval recall, paired on 9520 questions, k=64**

| tpf | distortion | avg recall | rrf |
|---|---|---|---|
| 252 | 1.61% | **33.37%** | 40.78% |
| 594 | 2.69% | 29.95% | 36.47% |
| 464 | 4.02% | 28.97% | 35.41% |

`252 -> 464` is **-4.40 pts, p = 5.7e-248**; `464 -> 594` recovers **+0.97, p = 2.9e-25** despite
28% more tokens. Ordering is exactly inverse to distortion.

**3. CG-Bench accuracy, full dataset, run off-cluster**

| budget | distortion | accuracy |
|---|---|---|
| 547-608 | 0.12% | **50.79%** |
| 482-536 | 0.71% | 50.60% |
| 600-640 | 2.69% | 50.44% |

Monotone in distortion, inverse to token count (578 > 509 > 640 tokens). Spread is only 0.35 pts,
so this replicates the *rank order*, not an effect size.

### Full-scale mechanism table (12,128 CG-Bench questions, all four arms paired)

Recall is the canonical `average_alignment_score` from `alignment_scores.csv`; GT contrast and
temporal bias are computed from the per-question dumps and merged on (video_id, question_index).

| tpf | distortion | GT contrast | recall | temporal bias r |
|---|---|---|---|---|
| 252 | 1.61% | **0.4485** | **32.71%** | **0.1021** |
| 510 | 0.71% | 0.3982 | 31.30% | 0.1070 |
| 594 | 2.69% | 0.3258 | 29.60% | 0.1475 |
| 464 | 4.02% | **0.2709** | **28.63%** | **0.2352** |

**Temporal bias is the most distortion-sensitive quantity measured.** Ordered by distortion
(0.71 < 1.61 < 2.69 < 4.02) the bias column is 0.1070, 0.1021, 0.1475, 0.2352 -- the two
low-distortion arms sit at ~0.10 regardless of carrying 252 or 510 tokens, while 464 has 2.3x
that. The swing is far larger than for contrast (1.65x) or recall (1.14x).

### Two channels, not one

    252 -> 510   (tokens x2, distortion 1.61% -> 0.71%)
        contrast -0.0503  p=3.9e-53      recall -1.40 pts  p=3.5e-36
        temporal bias +0.0049 -- essentially FLAT

    464 -> 510   (tokens +8%, distortion 4.02% -> 0.71%)
        contrast +0.1169  p=3.4e-158     recall +2.56 pts  p=9.4e-65
        and 510 recovers exactly the early-GT questions (see position table above)

So raw token count degrades contrast **without shifting where retrieval looks**, while grid
distortion drives the positional bias. These are separable mechanisms. 512 is anomalously bad
because it is the only budget where both fire at once -- high token count AND the worst grid.

NOTE on magnitudes: earlier single-video numbers (`BV1cF411o72x`, bias 0.316-0.546) are NOT
representative. At scale bias runs 0.10-0.24. That video is far more end-biased than typical;
treat its table as within-video ordering only.

## Caveats

- **Post-hoc.** The hypothesis was formed after seeing the 252 > 594 > 478 ordering. Three
  subsequent confirmations were predictions, but the origin still matters.
- **The low-token end does not fit.** 64 and 128 have the *highest* distortion (6.2%, 5.4%) and the
  *highest* GT contrast. Distortion cannot be the only factor; at low budgets something else
  dominates. Rank concordance over the four high-token budgets is 3/6 — chance — precisely because
  of them. The defensible claim is "two effects", not one.
- **LVBench accuracy differences are not individually significant** (478 -> 510 is +1.36, p=0.133;
  best pair is p=0.099). The recall evidence is the well-powered one.
- **GT contrast and temporal bias are correlated but NOT identical** -- partly resolved. The
  252-vs-510 pair has near-identical bias (0.1021 vs 0.1070) but clearly different contrast
  (0.4485 vs 0.3982), so contrast is not merely a restatement of the ramp. What remains untested
  is whether the *distortion* effect on contrast survives detrending block position before
  computing z; 464 has both the highest bias and the lowest contrast, so for that pair the two
  are still confounded.
- **Redundancy is NOT the channel**, at least as measured. Mean off-diagonal block self-similarity
  spans only 0.5754-0.5925 (3%) across all six budgets, against 42% for temporal bias. 512 is the
  max, consistent with the story, but the effect is far too small to explain the recall gap.

## Reproducing

Distortion needs no model run — it is arithmetic on video dimensions:

```python
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
rh, rw = smart_resize(h, w, factor=32, min_pixels=lo*1024, max_pixels=hi*1024)
distortion = abs(((rw//32)/(rh//32))/(w/h) - 1) * 100
```

Budgets live in `analysis/tpf_budgets.py`; `536` and `608` were added as low-distortion probes.
Recall comes from `analysis/compute_alignment_scores.py` (`--retrieval_k 64 --temporal_patch_size 2`
for Qwen3-VL; the default `temporal_patch_size=1` is wrong and makes recall look near-random).

## The decisive test (run 2026-07-27)

CG-Bench 536 vs 512 is the tightest available contrast: 510 vs 474 tokens/frame (+8%) but 0.71%
vs 4.02% distortion (5.7x). Token count and grid geometry predict opposite orderings.

Paired on 4,775 questions (`pk_results/.../64-0.5-tpf=482_536`, full 1,219-video dumps):

| tpf | distortion | GT contrast | recall |
|---|---|---|---|
| 252 | 1.61% | 0.4530 | 35.18% |
| **510** | **0.71%** | **0.4068** | **33.73%** |
| 594 | 2.69% | 0.3272 | 31.70% |
| 464 | 4.02% | 0.2899 | 31.16% |

    464 -> 510   contrast  +0.1169   p = 3.4e-158
    464 -> 510   recall    +2.56 pts p = 9.4e-65

**Distortion wins.** 510 carries 8% MORE tokens than 464 yet contrast is 40% higher and recall
2.56 points higher. Token count predicted the opposite.

But token count still does real work: 252 beats 510 on both measures despite 2.3x the distortion.
Ranked by contrast the order is 252 > 510 > 594 > 464; ranked by distortion it is 510 < 252 < 594
< 464. The two arms that break the token ordering are exactly the two low-distortion ones, so the
honest model is **two effects**, not one.

### Mechanism: it is the early-video questions

Recall by ground-truth position, 464 vs 510:

| GT position | n | 464 | 510 | delta |
|---|---|---|---|---|
| 0.0-0.1 | 1245 | 31.6% | 37.7% | **+6.2** |
| 0.1-0.3 | 1376 | 22.3% | 27.7% | **+5.3** |
| 0.3-0.7 | 1437 | 33.8% | 33.4% | -0.4 |
| 0.7-1.0 |  717 | 42.0% | 39.0% | -3.1 |

Fixing the grid recovers **exactly the early-video questions** that 512 loses, and gives a little
back at the end. That mirrors the 252 -> 464 degradation measured the same way (-8.86 pts for
GT position < 0.3, p=2.2e-246; **+1.72 pts** for position > 0.7, p=1.1e-07 -- opposite signs).

So the aggregate "recall degrades with tokens" is a **positional shift, not uniform degradation**:
retrieval moves toward the end of the video. The headline number is therefore sensitive to the
benchmark's GT-position distribution -- CG-Bench clues are front-loaded (2,621 questions in the
first 30% vs 716 in the last 30%), and a late-skewed benchmark would show the effect reversed.
That is worth stating before the effect is described as a general property.

Chain, end to end: **coarse grid quantization -> stronger temporal bias -> early-GT retrieval
collapse**, and correcting the grid reverses it without changing the token budget.

## Still open

- **The low-token end.** 64 and 128 are the most distorted budgets measured (6.32%, 5.43%) and
  cannot be otherwise: 9x16 = 144 is the SMALLEST exact 16:9 grid, so below it token count and
  distortion are confounded by construction. Budget 160 (144 tpf, 0.18%) is the control; arms for
  64/128/160 on CG-Bench were launched 2026-07-27.
- **Detrending.** GT contrast and temporal bias are still entangled; regressing out block position
  before computing z would separate cause from correlate.
