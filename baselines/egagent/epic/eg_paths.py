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
import threading
import time

R = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "egagent"))
for p in ("prepare_datasources", "egagent", ""):
    sys.path.insert(0, os.path.join(R, p) if p else R)

# arm -> the agent's backbone; the memory is the same for every arm. gpt41 and gemini25pro are the paper's;
# gemini36flash stands in for gemini25pro, which Google no longer serves to our key (README: Read first)
# Two arms are not the paper's at all and have a memory of their own, built by another model (run_egagent.sh sets
# EGAGENT_MEMORY_LLM): g5m-g51 (GPT-5-mini memory, GPT-5.1 agent) and q27 (Qwen3.8-27B for both, served locally)
BACKBONES = {"gpt41": "gpt-4.1", "gemini25pro": "gemini-2.5-pro", "gemini36flash": "gemini-3.6-flash",
             "g5m-g51": "gpt-5.1", "q27": "Qwen/Qwen3.8-27B"}
MEMORY_LLM = os.environ.get("EGAGENT_MEMORY_LLM") or "gpt-4.1"   # captions, graph extraction, relationship time stamps
EMBEDDER = "google/siglip2-giant-opt-patch16-384"    # frame / text-query embeddings (open weights, local)
CLIP_S = 30                                          # caption window (paper: 30 s clips at 1 fps)


# a returned model id that counts as the model asked for: the name itself or one of its dated snapshots
# (OpenRouter returns GPT-4.1's as openai/gpt-4.1... and Gemini's as google/..., an OpenAI-compatible gateway serving it from Azure OpenAI
# as azure/openai/gpt-4.1; the endpoint is recorded with each call)
SAME_MODEL = {"gpt-4.1": r"(azure/)?(openai/)?gpt-4\.1(-\d{4}-\d{2}-\d{2})?", "gemini-2.5-pro": r"(models/|google/)?gemini-2\.5-pro(-\d[\w.-]*)?",
              "gemini-3.6-flash": r"(models/|google/)?gemini-3\.6-flash(-\d[\w.-]*)?",
              "gpt-5-mini": r"(azure/|openai/)*gpt-5-mini(-\d{4}-\d{2}-\d{2})?", "gpt-5.1": r"(azure/|openai/)*gpt-5\.1(-\d{4}-\d{2}-\d{2})?",
              "Qwen/Qwen3.8-27B": r"Qwen/Qwen3\.8-27B"}
FIELDS = ("calls", "input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens")


def aliases(model):
    """Reply ids that also count as `model`: OPENAI_MODEL_ALIASES="gpt-4.1=<label>,...", for a gateway that answers
    with its deployment's own label (each call still records the label it returned)."""
    pairs = [x.split("=", 1) for x in os.environ.get("OPENAI_MODEL_ALIASES", "").split(",") if "=" in x]
    return {label.strip() for asked, label in pairs if asked.strip() == model}


class WrongModel(SystemExit):
    """Not an Exception on purpose: no retry loop (ours or LangChain's) may swallow it."""


FAILURE = []   # why this process failed, in one line: printed after the usage summary, so a log's tail shows it


def failed(why):
    """Set the failure line: an exception (its type, the HTTP status and error code an API error carries, the message)
    or a text."""
    if isinstance(why, BaseException):
        e, why = why, type(why).__name__ + ":"
        why += f" HTTP {e.status_code}" if getattr(e, "status_code", None) else ""
        why += f" [{e.code}]" if isinstance(getattr(e, "code", None), str) else ""
        why += f" {getattr(e, 'message', None) or e}"
    FAILURE[:] = [" ".join(str(why).split())[:220]]   # a board note keeps the last ~400 characters of a log


def watch_api(stage, model):
    """Watch every LLM call this process makes through LangChain: count calls and tokens per model id the API
    says answered, and stop the run (exit 4) when that id is not `model` (a provider may route a retired model's
    requests to another one). On exit one line is appended to $EGAGENT_USAGE (run_egagent.sh sums the lines of
    its stage, $EGAGENT_USAGE_STAGE, into runs.jsonl). A call whose reply names no model is counted under
    "unreported". output_tokens is what the API billed as output (Gemini's thinking included), cached_tokens and
    reasoning_tokens the parts of the input and the output it reported as such. A GPT row also has the
    `endpoint` that was called (a Gemini row too), and through OpenRouter the `provider` that served it and `cost`, the dollars
    OpenRouter charged, summed; `provider_note` is OPENAI_PROVIDER_NOTE (who serves the calls behind an
    OpenAI-compatible gateway, in words) and `system_fingerprints` the distinct fingerprints the replies carried.
    A process that fails prints `<stage> failed: <why>` after the summary (failed()). Calls from several threads
    (EGAGENT_QA_WORKERS) are counted under a lock."""
    from contextvars import ContextVar
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.tracers.context import register_configure_hook
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_openai import ChatOpenAI

    calls = {}
    lock = threading.Lock()
    if not hasattr(ChatOpenAI, "_egagent_result"):   # LangChain keeps the reply's usage but not who served it
        ChatOpenAI._egagent_result = ChatOpenAI._create_chat_result

        def result(self, response, *a, **kw):
            res = self._egagent_result(response, *a, **kw)
            extra = response if isinstance(response, dict) else getattr(response, "model_extra", None) or {}
            res.llm_output.update(endpoint=str(self.openai_api_base), provider=extra.get("provider"),
                                  provider_note=os.environ.get("OPENAI_PROVIDER_NOTE") or None)
            return res
        ChatOpenAI._create_chat_result = result
    if not hasattr(ChatGoogleGenerativeAI, "_egagent_generate"):   # the Gemini endpoint, as the client's transport has it
        ChatGoogleGenerativeAI._egagent_generate = ChatGoogleGenerativeAI._generate

        def generate(self, *a, **kw):
            res = self._egagent_generate(*a, **kw)
            res.llm_output = {**(res.llm_output or {}), "endpoint": str(self.client._transport._host)}
            return res
        ChatGoogleGenerativeAI._generate = generate

    class Watch(BaseCallbackHandler):
        raise_error = True
        run_inline = True

        def on_llm_end(self, response, **kw):
            with lock:
                self.count(response)

        def count(self, response):
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
            c.update({k: meta[k] for k in ("endpoint", "provider", "provider_note") if meta.get(k)})
            fp = meta.get("system_fingerprint")
            if fp and fp not in c.setdefault("system_fingerprints", []):
                c["system_fingerprints"].append(fp)
            cost = (meta.get("token_usage") or {}).get("cost")   # OpenRouter reports it with every reply
            if cost is not None:
                c["cost"] = round(c.get("cost", 0) + cost, 6)
            if sum(r["calls"] for r in calls.values()) % 25 == 0:   # a killed run's log still says what it spent
                print("API so far:", json.dumps(calls), flush=True)
            if got != "unreported" and not re.fullmatch(SAME_MODEL[model], got) and got not in aliases(model):
                print(f"STOP: asked {model}, the API answered with {got!r}; nothing from this call is kept", flush=True)
                failed(f"asked {model}, the API answered with {got!r} (exit 4)")
                raise WrongModel(4)

    var = ContextVar("egagent_api_watch", default=None)
    register_configure_hook(var, inheritable=True)
    var.set(Watch())
    hook = sys.excepthook   # an uncaught exception: the traceback as before, and its one line kept for the end

    def keep(t, e, tb):
        failed(e)
        hook(t, e, tb)
    sys.excepthook = keep

    def report():   # "stage" as run_egagent.sh names it (it sums the lines of a stage), "step" the script's own
        line = {"stage": os.environ.get("EGAGENT_USAGE_STAGE") or stage, "step": stage, "asked": model, "models": calls,
                "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        print("API calls:", json.dumps(line), flush=True)
        if calls and os.environ.get("EGAGENT_USAGE"):
            open(os.environ["EGAGENT_USAGE"], "a").write(json.dumps(line) + "\n")
        if FAILURE:
            print(f"{stage} failed: {FAILURE[0]}", flush=True)
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
