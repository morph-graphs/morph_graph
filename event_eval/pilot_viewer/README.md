# Event annotation viewer

Dynamic HTML viewer for comparing query-agnostic event annotations from
multiple VLMs on the same videos. One page per video: play the video, see each
model's events as a colored segment track, click a segment (or table row) to
read its description and jump the video to that moment.

## Run the viewer

**No dependencies, no API key** — just Python 3.8+ (any system `python3`)
and a browser. You need two things on disk: the annotation JSONL files and
the videos.

```bash
cd event_eval/pilot_viewer
python3 serve.py --port 8000
```

Open http://localhost:8000. If the machine is a remote cluster node, tunnel
from your laptop first:

```bash
ssh -L 8000:localhost:8000 <cluster-host>
```

### Configure (in the UI)

- **Video root** — the directory that contains the annotations' relative
  video paths. Annotations reference videos as `egolongqa/val/<id>.mp4`, so
  the root is whatever directory contains `egolongqa/` on *your* cluster.
- **Annotation JSONL files** — label + path pairs. Relative paths resolve
  against the repo root (`morph_graph/`); absolute paths work too. Hit
  **Load**: each file shows how many annotations it contains, then the video
  table lists per-model event counts.

Settings persist in the browser (localStorage).

### Using a video page

- Each model gets one horizontal track under the player; overlapping events
  stack into lanes. The vertical line is the playhead, synced to playback.
- Click a segment or table row: the description shows in the detail card and
  the video seeks to the event start. Hover a segment for a quick tooltip.
- Click empty track space to scrub to that time.


