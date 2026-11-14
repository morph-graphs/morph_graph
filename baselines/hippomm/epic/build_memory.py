#!/usr/bin/env python3
"""HippoMM's memory of one HD-EPIC video, with upstream's own code and settings (no questions needed).

--stage prep   upstream's frame extraction (adaptive, at most 1 frame/s, full resolution) and audio extraction
               (16 kHz mono; dropped when >90% silent) into the frame store. CPU only.
--stage build  upstream's HippocampalMemory.process_sequence: segmentation (SSIM / silence, 10-30 s), ImageBind
               features of every frame and of each segment's audio, Whisper (distil-large-v3) per segment,
               consolidation into one event (key frames by ImageBind similarity < 0.9), replay (one Qwen2.5-VL-7B
               caption per key frame, a one-sentence summary, Whisper over the whole audio). Then the event is
               written as the two memory files (hm_common.py). Needs a GPU and the Qwen2.5-VL endpoint.

    python build_memory.py --stage build --video <V> --video_file <V>.mp4 --store <work>/store --out <results dir>
    python build_memory.py --memory_files        # the memory's file name patterns, for run_hippomm.sh
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

import hm_common as hm


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--memory_files", action="store_true")
    ap.add_argument("--stage", choices=["prep", "build"])
    ap.add_argument("--video")
    ap.add_argument("--video_file")
    ap.add_argument("--store", help="upstream's memory_store folder (frames, audio, checkpoints): work files")
    ap.add_argument("--out", help="results folder of this video")
    a = ap.parse_args()
    if a.memory_files:
        print(" ".join(hm.MEMORY))
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    V, store = a.video, os.path.abspath(a.store)
    cfg = hm.config(store)
    from hippomm.core.batch_process import process_single_video

    video_file = Path(a.video_file)
    assert video_file.stem == V, "upstream takes the video id from the file name"
    res = process_single_video((video_file, Path(store), cfg))   # frames resume from their metadata.yaml
    if not res["success"]:
        sys.exit(f"extraction failed: {res['error']}")
    audio = res["audio_data"]
    print(f"# {V}: {len(res['frames'])} frames over {res['metadata']['duration']:.1f} s; audio "
          + (f"{len(audio) / res['sample_rate']:.1f} s" if audio is not None else "none (missing or >90% silent)"), flush=True)
    if a.stage == "prep":
        return

    from hippomm.core.hippocampal_memory import HippocampalMemory
    mem = HippocampalMemory(cfg)
    mem.add_video(V, res["metadata"])
    # segments -> features -> consolidate -> replay (resumes from upstream's checkpoint of the segments)
    mem.process_sequence(video_id=V, video_frames=res["frames"], frame_times=res["frame_times"],
                         audio_data=audio, audio_sample_rate=res["sample_rate"])
    if not mem.long_term_store:
        sys.exit("upstream produced no event (see the log above)")
    ev, segs = mem.long_term_store[-1], mem.short_term_buffer[V]
    bad = [c for c in ev.frame_captions if "[Error" in c]
    if bad:   # a caption request failed (endpoint down?): do not keep such a memory
        sys.exit(f"{len(bad)} of {len(ev.frame_captions)} frame captions failed, e.g. {bad[0]}")
    f = ev.features
    assert f["vision"].shape == (len(f["vision_times"]), 1024) and len(ev.frames) == len(ev.frame_captions) == len(ev.frame_times)
    n_audio = sum("audio" in m.modalities for m in segs)
    assert len(f.get("audio", [])) in (0, n_audio), "audio features do not line up with the segments"
    os.makedirs(a.out, exist_ok=True)
    jpath, fpath = hm.memory_paths(a.out, V)
    np.savez_compressed(fpath + ".tmp.npz", vision=f["vision"].astype(np.float32), vision_times=np.asarray(f["vision_times"], float),
                        audio=np.asarray(f.get("audio", np.zeros((0, 1024))), np.float32),
                        audio_times=np.asarray(f.get("audio_times", []), float))
    J = {"video": V, "duration": res["metadata"]["duration"], "settings": hm.settings(cfg),
         "frames": [os.path.relpath(p, store) for p in ev.frames], "frame_times": [float(t) for t in ev.frame_times],
         "frame_captions": ev.frame_captions,
         "segments": [{"start": float(m.segment_info.start_time), "end": float(m.segment_info.end_time),
                       "frames": len(m.segment_info.frames or []), "audio": "audio" in m.modalities,
                       "transcription": m.transcription} for m in segs],
         "holistic_audio_transcription": ev.holistic_audio_transcription, "summary": ev.summary}
    json.dump(J, open(jpath + ".tmp", "w"), indent=1)
    os.replace(fpath + ".tmp.npz", fpath)
    os.replace(jpath + ".tmp", jpath)
    print(f"# {V}: {len(f['vision'])} frame features, {len(ev.frames)} key frames with captions, {len(segs)} segments "
          f"({n_audio} with audio), {sum(len(s['transcription']) for s in J['segments'])} per-segment and "
          f"{len(ev.holistic_audio_transcription)} whole-audio Whisper lines -> {a.out}", flush=True)
    print(f"# summary: {ev.summary}", flush=True)


if __name__ == "__main__":
    main()
