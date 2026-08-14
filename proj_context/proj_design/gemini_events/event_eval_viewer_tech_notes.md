# Event-eval viewer — technical notes

Transferable lessons from building the HTML event viewer (scrubbing, caching,
remote access). Use this when you need similar tooling in other projects.

Related code: `morph_graph/event_eval/viewer/` (`server.py`, `static/app.js`).
Ops / CLI usage: `morph_graph/event_eval/README.md`.
Sibling design doc: `gemini_event_annotation_design.md`.

---

## 0. Launching the viewer (with portable video roots)

Annotation JSONs store **relative** `video_path`s with two parent directories
kept (e.g. `egolongqa/val/<id>.mp4`) so the same files work on any cluster.
The server resolves them against `--video-root` — the directory that
*contains* that relative path (absolute `video_path`s in older files still
work unchanged):

```bash
# from morph_graph/, any python3 ≥3.8 with ffmpeg on PATH
# (e.g. source .venv/bin/activate after `python -m event_eval.setup_ffmpeg`)
python -m event_eval.viewer \
  --video-root /lustre/fsw/portfolios/nemotron/users/pulkitk/tracking/datasets/egolongqa \
  --port 8765
```

Teammates on another cluster run the identical command with their own
`--video-root`. Annotation directories (oracle / agnostic / TSM / EventGraph)
default to `outputs/event_eval/*_latest` and are changeable in the UI —
relative dir inputs resolve against `morph_graph/`. Without `--video-root`,
relative video paths only resolve if they happen to exist under the CWD; the
sample still loads but shows "video missing".

From a laptop, tunnel to the host that runs the process (see §7):
`ssh -L 8765:127.0.0.1:8765 <host>` then open `http://127.0.0.1:8765/`.

---

## 1. Don’t rely on `<video>` for precise scrubbing

**Problem.** Browser `<video>` + remote/NFS mp4s often stall, flash black, or
seek inaccurately (keyframe-only seeks). Fine for casual playback; bad for
alignment / annotation / boundary QA.

**Pattern that works.** Treat the video as a **frame server**:

1. Probe duration + source fps (`ffprobe`).
2. Pick a **display / scrub fps** (often `1`, user-selectable).
3. Serve **JPEG stills** at times on that grid: `t = index / display_fps`.
4. UI is an `<img>` + range slider + ±step, not a media element.

This matches what many QC / annotation tools do (including seek-slider UIs
in Streamlit apps): time is discrete, images are cacheable, failures are
per-frame instead of whole-player freezes.

---

## 2. Scrub grid ≠ source fps

Keep two rates separate:

| Rate | Meaning |
|------|---------|
| `source_fps` | Native stream rate from `ffprobe` (metadata only) |
| `display_fps` | Sampling rate for scrub/play (default `1`) |

- Number of scrub steps ≈ `floor(duration * display_fps) + 1`
- Current time ≈ `frame_index / display_fps`
- Seeking from a timeline click: `index = round(t * display_fps)`

Changing `display_fps` rebuilds the grid; **preserve wall-clock time** when
possible (`t_keep`, then re-index).

---

## 3. Why naive “ffmpeg per click” feels terrible

Each cold request that runs:

```bash
ffmpeg -ss T -i video.mp4 -frames:v 1 ...
```

pays process startup + demux seek + decode + JPEG encode. On long ego videos
over NFS that is often **hundreds of ms to seconds** per step.

Fixes stack; use several at once:

1. **Downscale** for the UI (`scale='min(720,iw)':-2`) — decode/encode cost
   drops hard; annotation rarely needs native 4K.
2. **Cheaper JPEG** (`-q:v 5` is plenty for QA).
3. **Disk cache** keyed by `(video identity, display_fps, width, index)`.
4. **One-pass warm** of the whole strip at `display_fps` (below).
5. **Browser blob cache + neighbor prefetch** so ±1 steps hit memory.

---

## 4. Warm the strip in one ffmpeg pass

For a fixed scrub fps, prefer **one** filtergraph over N seeks:

```bash
ffmpeg -y -i video.mp4 -an -sn \
  -vf "fps=1,scale='min(720,iw)':-2" \
  -q:v 5 -start_number 0 \
  /cache/dir/%06d.jpg
```

- `fps=1` (or whatever `display_fps` is) samples on a regular grid.
- `-start_number 0` aligns filenames with `index = round(t * fps)`.
- Run this in a **background thread**; let single-frame extracts still serve
  the current step while warming finishes.
- UI can poll `cached/total` and show “warming…” → “cached”.

Video identity for cache dirs should include **path + mtime + size** so a
replaced file doesn’t reuse stale JPEGs.

Atomic writes for single-frame fills: write `*.partial.jpg`, then `replace()`
onto the final name (avoids half-written reads under concurrent requests).

---

## 5. Client-side caching pattern

```
ensureFrame(index)
  → memory Map(key → objectURL)? hit
  → else fetch /api/frame → blob → objectURL → Map
  → prefetch ±N neighbors (small concurrency before warm done)
```

Notes:

- Cap in-flight fetches (e.g. 2 while warming, higher after) so you don’t
  spawn a stampede of ffmpeg processes and make latency *worse*.
- `revokeObjectURL` when the video or `display_fps` changes.
- Debounce scrubber `input` so dragging doesn’t request every intermediate
  index; still update the playhead/time label immediately.

Keyboard: `,` / `←` prev, `.` / `→` next, `Space` play/pause — ignore when
focus is in inputs/textareas.

---

## 6. Stdlib HTTP server is enough for an internal viewer

This viewer uses `ThreadingHTTPServer` + static files + JSON APIs — no Flask.
Good for lab tools:

- Threading matters once frames warm: many concurrent GETs of JPEGs.
- Keep byte-range `/api/video` optional for debugging; don’t depend on it for UX.
- Absolute video paths from annotations are fine if the server resolves and
  checks existence/extension; don’t expose arbitrary filesystem reads without
  a allowlist/suffix check.

---

## 7. SSH tunnels: forward the machine that runs the process

Cluster layout (UMIACS-style):

- Login / Cursor host (e.g. `nexuscfar00`) ≠ GPU box (e.g. `legacygpu06`).
- GPU nodes often sit on **private `10.x`** — laptop can’t hit them directly.

If the viewer listens on `127.0.0.1:8765` **on the GPU box**:

```bash
# laptop → jump → GPU, forward local 8765 to GPU loopback
ssh -J user@nexuscfar00.umiacs.umd.edu \
  -L 8765:127.0.0.1:8765 \
  user@legacygpu06.umiacs.umd.edu
```

Then open `http://127.0.0.1:8765/` on the laptop.

**Common failure:** `-L 8765:127.0.0.1:8765` aimed at the *login* node while
the server runs on the GPU node — you forward the wrong loopback.

Alternatives:

- Cursor/VS Code **Remote SSH into the GPU host**, then use the Ports panel.
- Bind `--host 0.0.0.0` on the GPU box and
  `ssh -L 8765:legacygpu06:8765 user@nexuscfar00` — only works if the
  process listens on a reachable interface, not only `127.0.0.1`.

`127.0.0.1` on the server is correct and safer when you always ProxyJump
into that host.

---

## 8. Checklist for the next “frame scrubber” tool

1. Probe once; scrub on a discrete time grid.
2. Serve stills (JPEG/WebP), not a `<video>` playhead, for QA.
3. Downscale for UI; cache on disk by content identity.
4. Warm the grid with one `fps=…` ffmpeg pass; prefetch in the browser.
5. Expose scrub fps as a UI control; default low (1 fps) for long videos.
6. Tunnel to the **same host** that binds the port; confirm with
   `ss -ltnp | grep PORT` on that host.

---

## 9. Pointers in this repo

| Piece | Role |
|-------|------|
| `event_eval/viewer/server.py` | `probe_video`, `extract_frame_jpeg`, `start_warm_frames`, HTTP routes |
| `event_eval/viewer/static/app.js` | scrub grid, blob cache, prefetch, warm polling |
| `event_eval/viewer/static/index.html` | `<img>` + scrubber + FPS control |
| Temp cache | `$TMPDIR/event_eval_viewer_frames/<video_hash>/w<W>_fps<F>/######.jpg` |
