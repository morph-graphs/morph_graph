# Evidence-clips videos (`evclips_stitch.py`)

For the "evidence clips only" ablation: every ask_at "end" question of the seven `<bank>_end` banks (epic-morph
`v3/export_bank.py <bank>_end`, HD-EPIC videos) and of `stitched_eov` (`v3/export_bank.py stitched_eov`, the
kitchens' stitched videos) gets one short mp4 holding only its evidence clips, stitched in time order.

- **Spans**: the union of the question's `evidence.clips` ranges (dict or its str repr), merged where they overlap,
  no padding (`epic_eval.common.evidence_spans`; the cut at `t_ask` is the video's end for these questions),
  rounded to 0.01 s, then snapped to the source frame grid: frame k is shown at `v0 + k / fps` (`v0` the video
  stream's start time: 0 for HD-EPIC, 0.021 s for the stitched videos), frames `round((t0 - v0) * fps)` ..
  `round((t1 - v0) * fps) - 1`.
- **Cutting**: one input per span (seek half a frame before its first frame, `-t` = its frame count), re-encoded,
  so boundaries are frame-exact; concatenated at the source's constant frame rate.
- **Encoding**: MERIT's prep (`baselines/merit/epic/prepare_epic.py`): `scale=448:448`, no audio,
  `libx264 -preset veryfast -crf 23`, with the ffmpeg of MERIT's image.
- **Sharing**: questions of a video with the same spans share one file.

## Run

```bash
python baselines/common/evclips_stitch.py plan --bank_root <epic-morph>/outputs/v3/banks --video_dir <HD-EPIC videos> \
    --epic_morph_commit <commit> --extra_bank stitched_eov <bank root> <stitched videos> <commit> \
    --out OUT                                                       # OUT/plan.json; host python, no ffmpeg
OUT=OUT REPO=<checkout> sbatch --array=0-15 baselines/common/run_evclips_stitch.sbatch   # skip-existing, requeues
python baselines/common/evclips_stitch.py check --out OUT                                # streams of every file
python baselines/common/evclips_stitch.py check --out OUT --pixels --ids <id> ...         # frame-exact
python baselines/common/evclips_stitch.py manifest --out OUT                             # OUT/manifest.json
```

`stitch`, `check` and `manifest` run inside MERIT's image (ffmpeg, numpy).

## Layout and manifest

```
OUT/videos/<video>/<id>.mp4        the stitched file
OUT/segments/<video>/<id>.json     its record (the manifest's files[<id>])
OUT/manifest.json
```

`<id>` = first 12 hex of `sha1("<video>|<t0>-<t1>;<t0>-<t1>...")`, spans as `%.2f` seconds.

`manifest.json`:

- `encoder` (ffmpeg version line), `settings`, `source` (per bank: bank root, video folder, epic-morph commit and
  HEAD, whether `v3/export_bank.py` was modified, sha1 of every bank file), `n_questions`, `n_files`, `missing`.
- `questions`: `[{bank, video, key, qid, file}]`. `key` is the bank question's `key`, `qid` is
  `epic_eval.common.make_qid(video, q, t_ask)`, `file` is the `<id>`.
- `files`: `{<id>: {id, video, path, src, src_start_time, fps, frames, duration, bytes, encode_s, segments}}`. `path`
  is relative to the manifest, `src` the source video it was cut from, `src_start_time` its video stream's `v0`. `fps` is a fraction string (`"30/1"`). `segments` is
  `[{src_start, src_end, stitched_start, stitched_end, src_frames: [k0, k1], span: [t0, t1]}]`, in seconds; the
  times are frame-exact (`k / fps`), the end is exclusive, and `span` is the requested range before snapping.
  A stitched time `t` in segment `s` is source time `s.src_start + (t - s.stitched_start)`.
