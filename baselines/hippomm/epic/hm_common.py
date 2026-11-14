"""Shared by the HippoMM adapter scripts: the upstream code on sys.path, its config, the memory files of one
video, and the memory as it stood at a question's cutoff.

The memory of one video (results folder; all of it is independent of the questions):
  {video}_memory.json  key frames (paths relative to the frame store, times, Qwen2.5-VL captions), HippoMM's
                       segments (start, end, per-segment Whisper text), the whole-audio Whisper transcription,
                       the whole-video summary, and the settings it was built with
  {video}_features.npz ImageBind features: vision (every extracted frame) and audio (one per segment), with times
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("HIPPOMM_SRC") or os.path.join(HERE, "..", "HippoMM")   # the upstream checkout
sys.path.insert(0, SRC)

VLM = "Qwen/Qwen2.5-VL-7B-Instruct"     # frame captions and summaries (served locally)
REASONING = "gpt-4o"                    # upstream's reasoning model (OpenAI API)
MEMORY = ["{video}_memory.json", "{video}_features.npz"]
# upstream's summary prompts (HippocampalMemory.replay), used again for the summary at a question's cutoff
SUMMARY_PROMPT = {
    (True, True): "Please provide a concise one sentence summary of this event based on the video frames descriptions and audio transcription. What is happening in this event?",
    (True, False): "Please provide a concise one sentence summary of this event based on the video frames descriptions. What is happening in this event?",
}


def memory_paths(out, video):
    return [os.path.join(out, m.format(video=video)) for m in MEMORY]


def config(store, qwen_url=None):
    """Upstream's default config with our paths. The OpenAI key is read from the environment here and only
    lives in this dict (never in a file or on a command line)."""
    import yaml
    c = yaml.safe_load(open(os.path.join(SRC, "config", "default_config.yaml")))
    c["storage"]["base_dir"] = store
    c["models"]["imagebind_path"] = os.environ.get("HIPPOMM_IMAGEBIND", c["models"]["imagebind_path"])
    c["models"]["qwen_path"] = VLM
    url = qwen_url or os.environ.get("HIPPOMM_VLM_URL", "http://127.0.0.1:8000/v1")
    c["api"]["qwen"] = {"base_url": url, "api_key": "EMPTY", "model_name": VLM}
    c["api"]["frame_processing"] = {"base_urls": [url], "api_key": "EMPTY"}
    c["api"]["reasoning"] = {"api_key": os.environ.get("OPENAI_API_KEY", ""), "model_name": REASONING}
    return c


def settings(c):
    """What the memory depends on besides the video (recorded in the memory file)."""
    return {"vlm": VLM, "whisper": c["models"]["whisper_model"], "imagebind": "imagebind_huge",
            "processing": c["processing"]}


def load_memory(out, video):
    j, f = memory_paths(out, video)
    return json.load(open(j)), np.load(f)


def cutoff(q, video_end):
    """A streaming question sees the video up to its t_ask; any other question sees all of it."""
    return float(q["t_ask"]) if q.get("streaming") and "t_ask" in q else float(video_end)


def at_cutoff(J, F, cut):
    """The memory restricted to what HippoMM would hold at time `cut`: frames (features, key frames, captions)
    seen by then, segments (audio feature, Whisper text) that ended by then, whole-audio transcription lines
    that ended by then. Key-frame selection is greedy in time order, so the key frames before `cut` are the
    ones a build of the first `cut` seconds selects."""
    nv = int((F["vision_times"] <= cut).sum())
    nk = int(sum(t <= cut for t in J["frame_times"]))
    segs = [s for s in J["segments"] if s["end"] <= cut]
    # upstream's consolidation keeps a segment's audio feature only when the segment's content has an 'audio'
    # entry, which process_sequence never sets: the event then has no audio features (F["audio"] is empty)
    na = sum(s["audio"] for s in segs) if len(F["audio"]) else 0
    return {"nv": nv, "nk": nk, "na": na,
            "captions": J["frame_captions"][:nk],
            "transcription": [t for s in segs for t in s["transcription"] or []],
            "holistic": [t for t in J["holistic_audio_transcription"] if t["end"] <= cut]}


def theta_event(J, F, cut, summary, store):
    """Upstream's ThetaEvent for the memory at `cut` (what QARecallSystem reads from long_term_store)."""
    from hippomm.core.hippocampal_memory import ThetaEvent
    m = at_cutoff(J, F, cut)
    feats, times = {"vision": F["vision"][:m["nv"]]}, {"vision_times": F["vision_times"][:m["nv"]]}
    if m["na"]:
        feats["audio"], times["audio_times"] = F["audio"][:m["na"]], F["audio_times"][:m["na"]]
    return ThetaEvent(
        features=feats, feature_times=times,
        frames=[os.path.join(store, p) for p in J["frames"][:m["nk"]]], frame_times=J["frame_times"][:m["nk"]],
        frame_captions=m["captions"], audio_times=[float(t) for t in F["audio_times"][:m["na"]]],
        audio_transcription=m["transcription"], holistic_audio_transcription=m["holistic"],
        summary=summary, start_time=0.0, end_time=cut)
