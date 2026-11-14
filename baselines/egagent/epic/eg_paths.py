"""Import first in every epic/ script that uses EGAgent: puts the upstream code on sys.path and holds the few
things the scripts share (memory file names, time formats, the paper's models). Imports nothing heavy.

Upstream is not installed; its modules are the clone's root (paths, utils, retrieval_model), egagent/ and
prepare_datasources/. Its settings come from the environment (our patch, ../epic_morph.patch):
EGAGENT_DATASET, EGAGENT_BACKBONE, EGAGENT_DB_ROOT, EGAGENT_MODEL_ROOT, OPENAI_API_KEY (or OPENROUTER_API_KEY alone:
the GPT calls through OpenRouter, README "Through OpenRouter"), GEMINI_API_KEY.
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
# (OpenRouter returns GPT-4.1's as openai/gpt-4.1...)
SAME_MODEL = {"gpt-4.1": r"(openai/)?gpt-4\.1(-\d{4}-\d{2}-\d{2})?", "gemini-2.5-pro": r"(models/)?gemini-2\.5-pro(-\d[\w.-]*)?"}
FIELDS = ("calls", "input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens")


class WrongModel(SystemExit):
    """Not an Exception on purpose: no retry loop (ours or LangChain's) may swallow it."""


def watch_api(stage, model):
    """Watch every LLM call this process makes through LangChain: count calls and tokens per model id the API
    says answered, and stop the run (exit 4) when that id is not `model` (a provider may route a retired model's
    requests to another one). On exit one line is appended to $EGAGENT_USAGE (run_egagent.sh sums the lines of
    its stage, $EGAGENT_USAGE_STAGE, into runs.jsonl). A call whose reply names no model is counted under "unreported". output_tokens is what the API
    billed as output (Gemini's thinking included), cached_tokens and reasoning_tokens the parts of the input and
    the output it reported as such. A GPT row also has the `endpoint` that was called, and through OpenRouter
    the `provider` that served it and `cost`, the dollars OpenRouter charged, summed."""
    from contextvars import ContextVar
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.tracers.context import register_configure_hook
    from langchain_openai import ChatOpenAI

    calls = {}
    if not hasattr(ChatOpenAI, "_egagent_result"):   # LangChain keeps the reply's usage but not who served it
        ChatOpenAI._egagent_result = ChatOpenAI._create_chat_result

        def result(self, response, *a, **kw):
            res = self._egagent_result(response, *a, **kw)
            extra = response if isinstance(response, dict) else getattr(response, "model_extra", None) or {}
            res.llm_output.update(endpoint=str(self.openai_api_base), provider=extra.get("provider"))
            return res
        ChatOpenAI._create_chat_result = result

    class Watch(BaseCallbackHandler):
        raise_error = True
        run_inline = True

        def on_llm_end(self, response, **kw):
            msg = getattr(response.generations[0][0], "message", None)
            meta = getattr(msg, "response_metadata", None) or {}
            got = meta.get("model_name") or meta.get("model_version") or meta.get("model") or "unreported"
            use = getattr(msg, "usage_metadata", None) or {}
            c = calls.setdefault(got, dict.fromkeys(FIELDS, 0))
            c["calls"] += 1
            c["input_tokens"] += use.get("input_tokens") or 0
            c["cached_tokens"] += (use.get("input_token_details") or {}).get("cache_read") or 0
            c["output_tokens"] += use.get("output_tokens") or 0
            c["reasoning_tokens"] += (use.get("output_token_details") or {}).get("reasoning") or 0
            c.update({k: meta[k] for k in ("endpoint", "provider") if meta.get(k)})
            cost = (meta.get("token_usage") or {}).get("cost")   # OpenRouter reports it with every reply
            if cost is not None:
                c["cost"] = round(c.get("cost", 0) + cost, 6)
            if sum(r["calls"] for r in calls.values()) % 25 == 0:   # a killed run's log still says what it spent
                print("API so far:", json.dumps(calls), flush=True)
            if got != "unreported" and not re.fullmatch(SAME_MODEL[model], got):
                print(f"STOP: asked {model}, the API answered with {got!r}; nothing from this call is kept", flush=True)
                raise WrongModel(4)

    var = ContextVar("egagent_api_watch", default=None)
    register_configure_hook(var, inheritable=True)
    var.set(Watch())

    def report():   # "stage" as run_egagent.sh names it (it sums the lines of a stage), "step" the script's own
        line = {"stage": os.environ.get("EGAGENT_USAGE_STAGE") or stage, "step": stage, "asked": model, "models": calls,
                "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
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
