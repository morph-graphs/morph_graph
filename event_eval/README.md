# Query-Conditioned Event Evaluation

This package evaluates query-agnostic TSM segments against a
query-conditioned Gemini event oracle. It is evaluation-only; it does not
define training targets for the final query-agnostic memory.

## Outputs

- Gemini oracle: minimal meaningful events needed to answer one question
- TSM export: query-agnostic boundaries, spans, novelty, and exact timestamps
- comparison: TSM vs **query-agnostic** Gemini — boundary error,
  over-/under-segmentation, contamination, and tolerance-based boundary F1

Mere object visibility is not treated as an event. Objects participate in
events only through intentional actions, interactions, state changes, or
meaningful context transitions.

## Environment

```bash
module load ffmpeg/7.1
source /fs/cfar-projects/mem_open_vocab/miniconda3/bin/activate morph_graphs_qwen3_cu126
pip install google-genai
export GOOGLE_API_KEY=...
```

For the Kimi (Moonshot) provider instead of Gemini:

```bash
pip install openai  # already satisfied in the shared conda env
export MOONSHOT_API_KEY=...
```

Run all commands from the `morph_graph/` directory.

## 1. Annotate a pilot

Create a deterministic query-structure-stratified pilot manifest. The selector
uses the dashboard's evidence-scope, link-basis, and answer-operation taxonomy;
it no longer uses the old keyword association tags:

```bash
python -m event_eval.select_pilot \
  --count 8 \
  --output-json outputs/event_eval/pilot_ids.json
```

Then run query-conditioned oracles (one per question):

```bash
python -m event_eval.annotate \
  --benchmark egolongqa \
  --mode query_conditioned \
  --sample-ids-json outputs/event_eval/pilot_ids.json \
  --output-dir outputs/event_eval/oracles
```

And query-agnostic Gemini segmentations (one per unique video). Uses prompt
`eval_agnostic_v2` (compact activity-level partitions; hierarchy rare). Re-run
with `--overwrite` after prompt changes:

```bash
python -m event_eval.annotate \
  --benchmark egolongqa \
  --mode query_agnostic \
  --sample-ids-json outputs/event_eval/pilot_ids.json \
  --output-dir outputs/event_eval/agnostic \
  --overwrite
```

The runner stores one validated JSON file per job, skips valid existing
outputs, and writes `.error.json` records for failures (including the last
`raw_response` / `parsed_events` when Gemini returned a payload that failed
JSON parse or schema validation). It uses `gemini-3.1-pro-preview`, 1 FPS,
medium media resolution, deterministic structured output, and
`--max-output-tokens 32768` by default (avoids truncated JSON on dense
agnostic segmentations). Agnostic outputs are keyed by `video_id`.

Use `--max-samples N` for an inexpensive smoke test. Do not pass API keys on
the command line; use `GOOGLE_API_KEY`.

### Kimi (Moonshot) provider

To pilot Kimi K2.5 as a cheaper alternative to Gemini, pass `--provider kimi`
to either annotation command above, e.g.:

```bash
python -m event_eval.annotate \
  --benchmark egolongqa \
  --mode query_agnostic \
  --provider kimi \
  --sample-ids-json outputs/event_eval/pilot_ids.json \
  --output-dir outputs/event_eval/agnostic_kimi \
  --overwrite
```

Uses model `kimi-k2.5` by default (override with `--model`), Moonshot's
hosted API at `https://api.moonshot.ai/v1` (override with `--base-url`, e.g.
for the `.cn` endpoint), and the same prompts/schema/output format as the
Gemini path, so `metrics.py`/`report.py` and the viewer work unchanged and
Kimi vs. Gemini outputs are directly comparable. `--fps` and
`--media-resolution` are Gemini-only (Kimi's API exposes no equivalent
sampling control); the saved `video_fps` field for Kimi outputs is a nominal
placeholder, not a guaranteed sample rate. Retry/output-token flags
(`--max-retries`, `--retry-delay-sec`, `--max-output-tokens`) apply to both
providers.

Self-hosting Kimi K2.5 (vs. the hosted API used here) needs ~8x H200 GPUs per
Moonshot's own deployment guide; not attempted by this module yet.

## 2. Export TSM for oracle videos

Run this step on a GPU node:

```bash
module load cuda/12.6.3
module load ffmpeg/7.1

python -m event_eval.tsm \
  --oracle-dir outputs/event_eval/oracles \
  --output outputs/event_eval/tsm
```

The exporter runs each unique video once. It preserves the current TSM
defaults and maps temporal units to seconds using sampled source-frame indices,
which remains correct when the frame cap reduces the effective sampling rate.

## 3. Compare (TSM vs query-agnostic)

Primary segmentation metrics compare TSM to **query-agnostic** Gemini (one
comparison per video), not to the query-conditioned oracle:

```bash
python -m event_eval.report \
  --agnostic-dir outputs/event_eval/agnostic \
  --tsm-dir outputs/event_eval/tsm \
  --output-json outputs/event_eval/report.json
```

Legacy conditioned comparison (per question) still works with `--oracle-dir`
instead of `--agnostic-dir` (pass exactly one).

For a single pair (agnostic or oracle JSON):

```bash
python -m event_eval.metrics \
  --oracle-json outputs/event_eval/agnostic/VIDEO_ID.json \
  --tsm-json outputs/event_eval/tsm/VIDEO_ID.json \
  --output-json outputs/event_eval/comparison.json
```

## Metric interpretation

Reference events below are **query-agnostic** Gemini spans (unless you passed
`--oracle-dir`).

- Boundary error: distance between a reference event's start/end and its
  minimal covering TSM segments.
- Over-segmentation: mean extra TSM segments per reference event,
  `max(0, n_tsm_intersecting - 1)`.
- Under-segmentation: mean extra reference events per TSM segment that hits
  any reference, `max(0, n_ref_intersecting - 1)`.
- Contamination: footage outside the reference event included by those
  segments.
- Boundary F1: one-to-one matching at absolute tolerances (prefer @2s).

Query-event coverage is intentionally deferred to retrieval evaluation. The
complete TSM partition covers the whole video, so segmentation-only coverage
would always be 100%.

## 4. HTML event viewer

Browse oracle / query-agnostic Gemini / TSM on the actual videos (frame-by-frame
scrubbing via ffmpeg JPEGs, triple timeline, parent→child edges, per-event
cover metrics). Paths are preset to the pilot outputs but can be edited in the
UI.

For reusable design notes (why not `<video>`, ffmpeg warm/cache, SSH tunnels),
see
[`proj_context/proj_design/gemini_events/event_eval_viewer_tech_notes.md`](../proj_context/proj_design/gemini_events/event_eval_viewer_tech_notes.md).

```bash
python -m event_eval.viewer
# optional: --host 0.0.0.0 --port 8765
```

Open `http://127.0.0.1:8765/`. Defaults:

- `outputs/event_eval/oracles` (query-conditioned Gemini)
- `outputs/event_eval/agnostic` (query-agnostic Gemini)
- `outputs/event_eval/tsm`
- `outputs/event_eval/report.json`
- `data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl`
  (ground-truth answers)

Frames are extracted via `ffmpeg` as downscaled JPEGs (default width 720),
disk-cached under the system temp dir. On sample load the server warms the
whole scrub-FPS strip in one background pass so stepping becomes near-instant
once warming finishes; the UI also prefetches nearby steps. Requires
`module load ffmpeg` (or ffmpeg on PATH). No extra Python deps. Scrub/play
defaults to **1 FPS** (changeable in the UI). Keys: `,` / `←` prev step,
`.` / `→` next step, `Space` play/pause.

## Tests

```bash
python -m unittest tests.test_event_eval
```

