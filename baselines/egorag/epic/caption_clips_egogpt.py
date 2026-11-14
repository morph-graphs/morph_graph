#!/usr/bin/env python3
"""EgoRAG's memory, step 1: one caption per 30 s clip with EgoGPT, the paper's captioner (audio + video).

Uses upstream's own wrapper (EgoRAG/egorag/models/egogpt.py: 1 frame per second, greedy, up to 4096 new
tokens) and the caption prompt of EgoRAG/main.py. One change: upstream's wrapper slices the audio with the
HHMMSSFF clock integers as if they were seconds, so every clip gets an empty slice, i.e. silence. The paper
captions from "both visual and audio inputs", so here the clip's own audio track is passed (--audio silent
gives the released code's behaviour).

Reads <subject>_30sec_skeleton.json (prepare_clips.py), writes <subject>_30sec.json in the format of
EgoRAG's create_database.py (start_time, end_time, text, date, video_path). Resumes: clips with text are skipped.
"""
import argparse
import json
import os
import time

import librosa
import numpy as np
from huggingface_hub import snapshot_download

# EgoRAG/main.py, create stage (its system message is dropped by upstream's EgoGPT wrapper, so it is here too)
PROMPT = ("Imagine you are the character in the video, describing from a first-person perspective what you saw, "
          "and everything that happened over time. Use I as the subject.")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--skeleton", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--clip_root", required=True)
    p.add_argument("--model", default="lmms-lab/EgoGPT-7b-EgoIT-EgoLife", help="HF id or a local copy (EgoIT-99K + EgoLife day 1)")
    p.add_argument("--audio", choices=["clip", "silent"], default="clip")
    p.add_argument("--max_clips", type=int, default=None, help="caption only the first N uncaptioned clips (smoke test)")
    a = p.parse_args()
    a.skeleton, a.output, a.clip_root = map(os.path.abspath, (a.skeleton, a.output, a.clip_root))

    clips = json.load(open(a.skeleton))
    if os.path.exists(a.output):
        done = {c["start_time"]: c.get("text") for c in json.load(open(a.output))}
        for c in clips:
            if done.get(c["start_time"]):
                c["text"] = done[c["start_time"]]
    todo = [c for c in clips if not c.get("text")][: a.max_clips]
    print(f"{len(clips)} clips, {len(todo)} to caption", flush=True)
    if not todo:
        return

    import egogpt.model.builder   # noqa: F401  upstream's wrapper hides a failed EgoGPT import behind a print
    from egorag.models.egogpt import EgoGPT   # needs EgoLife/EgoGPT and EgoLife/EgoRAG on PYTHONPATH

    class EgoGPTClip(EgoGPT):
        def process_video(self, video_path, video_start_time, start_time, end_time, fps=1):
            d = super().process_video(video_path, video_start_time, start_time, end_time, fps)
            if a.audio == "clip":
                d["audio"], _ = librosa.load(video_path, sr=16000)   # the clip file is the 30 s segment
            assert a.audio == "clip" or len(d["audio"]) == 0, "upstream's audio slice is no longer empty"
            return d

    # the checkpoint's config names its Whisper encoder as ./large-v3.pt: load from the folder that holds it
    path = a.model if os.path.isdir(a.model) else snapshot_download(a.model)
    os.chdir(os.path.join(path, "speech_encoder"))
    model = EgoGPTClip(path)
    for n, c in enumerate(todo, 1):
        t0 = time.time()
        f = os.path.join(a.clip_root, c["video_path"])
        # the clip file starts at the clip's start time, so video_start_time = start_time
        c["text"] = model.inference_video(f, int(c["start_time"]), int(c["start_time"]), int(c["end_time"]), PROMPT).strip()
        assert c["text"], f"empty caption for {f}"
        print(f"[{n}/{len(todo)}] {c['start_time']} {time.time() - t0:.1f}s: {c['text'][:160]}", flush=True)
        json.dump(clips, open(a.output + ".tmp", "w"), indent=2, ensure_ascii=False)
        os.replace(a.output + ".tmp", a.output)
    words = [len(c["text"].split()) for c in clips if c.get("text")]
    import torch
    print(f"captioned {len(words)}/{len(clips)} clips, words per caption mean {np.mean(words):.0f} max {max(words)}; "
          f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")


if __name__ == "__main__":
    main()
