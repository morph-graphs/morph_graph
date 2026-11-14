"""Import first in every epic/ script that uses EGAgent: puts the upstream code on sys.path and holds the few
things the scripts share (memory file names, time formats, the paper's models). Imports nothing heavy.

Upstream is not installed; its modules are the clone's root (paths, utils, retrieval_model), egagent/ and
prepare_datasources/. Its settings come from the environment (our patch, ../epic_morph.patch):
EGAGENT_DATASET, EGAGENT_BACKBONE, EGAGENT_DB_ROOT, EGAGENT_MODEL_ROOT, OPENAI_API_KEY, GEMINI_API_KEY.
"""
import atexit
import json
import os
import re
import sys
import time

R = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "egagent"))
for p in ("prepare_datasources", "egagent", ""):
    sys.path.insert(0, os.path.join(R, p) if p else R)

# the paper's configuration: arm -> the agent's backbone; the memory is the same for every arm
BACKBONES = {"gpt41": "gpt-4.1", "gemini25pro": "gemini-2.5-pro"}
MEMORY_LLM = "gpt-4.1"                               # captions, graph extraction, relationship time stamps
EMBEDDER = "google/siglip2-giant-opt-patch16-384"    # frame / text-query embeddings (open weights, local)
CLIP_S = 30                                          # caption window (paper: 30 s clips at 1 fps)


# a returned model id that counts as the model asked for: the name itself or one of its dated snapshots
SAME_MODEL = {"gpt-4.1": r"gpt-4\.1(-\d{4}-\d{2}-\d{2})?", "gemini-2.5-pro": r"(models/)?gemini-2\.5-pro(-\d[\w.-]*)?"}


class WrongModel(SystemExit):
    """Not an Exception on purpose: no retry loop (ours or LangChain's) may swallow it."""


def watch_api(stage, model):
    """Watch every LLM call this process makes through LangChain: count calls and tokens per model id the API
    says answered, and stop the run (exit 4) when that id is not `model` (a provider may route a retired model's
    requests to another one). On exit one line is appended to $EGAGENT_USAGE (run_egagent.sh sums them into
    runs.jsonl). A call whose reply names no model is counted under "unreported"."""
    from contextvars import ContextVar
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.tracers.context import register_configure_hook

    calls = {}

    class Watch(BaseCallbackHandler):
        raise_error = True
        run_inline = True

        def on_llm_end(self, response, **kw):
            msg = getattr(response.generations[0][0], "message", None)
            meta = getattr(msg, "response_metadata", None) or {}
            got = meta.get("model_name") or meta.get("model_version") or meta.get("model") or "unreported"
            use = getattr(msg, "usage_metadata", None) or {}
            c = calls.setdefault(got, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
            c["calls"] += 1
            c["input_tokens"] += use.get("input_tokens") or 0
            c["output_tokens"] += use.get("output_tokens") or 0
            if got != "unreported" and not re.fullmatch(SAME_MODEL[model], got):
                print(f"STOP: asked {model}, the API answered with {got!r}; nothing from this call is kept", flush=True)
                raise WrongModel(4)

    var = ContextVar("egagent_api_watch", default=None)
    register_configure_hook(var, inheritable=True)
    var.set(Watch())

    def report():
        line = {"stage": stage, "asked": model, "models": calls, "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        print("API calls:", json.dumps(line), flush=True)
        if calls and os.environ.get("EGAGENT_USAGE"):
            open(os.environ["EGAGENT_USAGE"], "a").write(json.dumps(line) + "\n")
    atexit.register(report)
    return calls


def memory_files(video):
    """The memory of one video, as flat files of its results folder (what `upload` shares)."""
    return [f"{video}_frames.npy", f"{video}_captions.json", f"{video}_graph.json"]


def hhmmss(sec):
    sec = int(sec)
    return f"{sec // 3600:02d}{sec % 3600 // 60:02d}{sec % 60:02d}"


def clock(sec, ms=False):
    s = hhmmss(sec)
    return f"{s[:2]}:{s[2:4]}:{s[4:]}" + (",000" if ms else "")


def frame_files(frames_dir):
    """The 1 fps frames in time order: frame i (0-based) is named for second i + 1, the time it is complete."""
    return sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
