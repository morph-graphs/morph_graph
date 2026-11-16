"""The evidence-clips videos (the "evidence clips only" ablation; made by baselines/common/evclips_stitch.py): where a
question's stitched video is, what it holds, and its frame times in the source video.

A stitched video is one question's evidence spans (the union of its bank evidence.clips ranges, merged, in time order,
no padding) cut from the source and joined; questions of a video with the same spans share it. The stitcher's output
folder, and the Hugging Face dataset made from it (EVCLIPS_HF_REPO, default morph-graphs/epic-morph-evclips), hold
manifest.json:
    {"questions": [{"bank", "video", "key", "qid", "file"}, ...],
     "files": {<file id>: {"video", "path" (relative to the folder), "fps", "frames", "duration",
                          "segments": [{"src_start", "src_end", "stitched_start", "stitched_end", "span"}, ...]}}}
and the videos at their `path`. EVCLIPS_DIR, when set, is such a folder on this machine (read first); otherwise the
manifest and a video's files come from the dataset (into the Hugging Face cache), only the ones asked for.
"""
import functools
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from epic_eval.common import evidence_spans  # noqa: E402

REPO = os.environ.get("EVCLIPS_HF_REPO", "morph-graphs/epic-morph-evclips")


def _get(rel):
    """A file of the folder: EVCLIPS_DIR's when it has it, else the dataset's (downloaded once into the cache)."""
    local = os.environ.get("EVCLIPS_DIR")
    if local and os.path.exists(os.path.join(local, rel)):
        return os.path.join(local, rel)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO, rel, repo_type="dataset")


@functools.lru_cache(maxsize=None)
def manifest():
    return json.load(open(_get("manifest.json")))


def lookup(video, key, clips, t_ask):
    """The stitched video of a question (its bank key, evidence clips and t_ask): the manifest's file entry, with
    `id` and `file` (a local path). Stops when the question is not in the manifest or the bank's spans have moved
    since the video was stitched (rounded to 0.01 s, as the stitcher rounds them)."""
    m = manifest()
    fid = {(q["video"], q["key"]): q["file"] for q in m["questions"]}.get((video, key))
    if fid is None:
        raise SystemExit(f"{video} {key}: not in the evidence-clips manifest ({REPO} or EVCLIPS_DIR)")
    rec = dict(m["files"][fid], id=fid)
    spans = [[round(a, 2), round(b, 2)] for a, b in evidence_spans({"qid": key, "t_ask": t_ask, "evidence_clips": clips})]
    assert [s["span"] for s in rec["segments"]] == spans, f"{video} {key}: bank spans {spans}, stitched {rec['segments']}"
    rec["file"] = _get(rec["path"])
    return rec


def to_source(rec, t):
    """The source-video time of time t in the stitched video (the segment it falls in; past the end, the end)."""
    seg = next((s for s in rec["segments"] if t < s["stitched_end"]), rec["segments"][-1])
    return round(seg["src_start"] + min(t, seg["stitched_end"]) - seg["stitched_start"], 3)


def frames_1fps(rec, root, size=448):
    """[(jpg, time in the stitched video)] at 1 fps and size x size, cut as MERIT's prep cuts a video's frames
    (prepare_epic.make_frames: fps=1, scale, -q:v 3): frame k is the picture at k + 0.5 s (Direct's measured rule).
    ffmpeg also writes a frame k whose k + 0.5 is past the video's end (a repeat of its last picture): left out, so
    every frame shown has its own time (at least one). Cached in root/<video>/<file id>/."""
    d = os.path.join(root, rec["video"], rec["id"])
    if not os.path.exists(os.path.join(d, "_done")):
        os.makedirs(os.path.dirname(d), exist_ok=True)
        tmp = tempfile.mkdtemp(prefix=rec["id"] + ".", dir=os.path.dirname(d))   # one per call: threads, containers
        os.chmod(tmp, 0o775 & ~os.umask(os.umask(0)))   # mkdtemp's 0700, widened as the work folder's umask allows
        subprocess.run(["ffmpeg", "-loglevel", "error", "-i", rec["file"], "-vf", f"fps=1,scale={size}:{size}",
                        "-q:v", "3", "-start_number", "0", os.path.join(tmp, "t%05d.jpg")], check=True)
        open(os.path.join(tmp, "_done"), "w").close()
        if os.path.exists(d) and not os.path.exists(os.path.join(d, "_done")):
            shutil.rmtree(d, ignore_errors=True)   # an interrupted earlier cut
        try:
            os.rename(tmp, d)
        except OSError:   # another process cut it first: theirs is the same
            shutil.rmtree(tmp, ignore_errors=True)
        assert os.path.exists(os.path.join(d, "_done")), d
    jpgs = sorted(f for f in os.listdir(d) if f.endswith(".jpg"))
    out = [(os.path.join(d, f), k + 0.5) for k, f in enumerate(jpgs) if k + 0.5 <= rec["duration"]]
    return out or [(os.path.join(d, jpgs[0]), min(0.5, rec["duration"]))]
