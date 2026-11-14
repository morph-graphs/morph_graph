"""The official arms' OpenAI calls, counted (import after wmm_paths in every epic/ script that may make them).

WorldMM's OpenAIModel returns only the reply text. Here every OpenAIModel gets its Responses calls wrapped, so
each real API call (not a hit of WorldMM's sqlite reply cache) adds the tokens the API billed and the model id it
answered with to WORLDMM_USAGE_FILE, under the pipeline step set with stage() or the model's `usage_role`:

    {"captions": {"gpt-5-mini-2025-08-07": {"calls": 163, "input_tokens": ..., "cached_tokens": ...,
                                            "output_tokens": ..., "reasoning_tokens": ..., "service_tier": "default"}}, ...}

An empty reply (no text and nothing parsed) is counted, since it was billed, and then raised as an error: WorldMM's
retry asks again, and its reply cache never stores it (a cached empty reply would be served on every rerun).
The file is cumulative: a resumed stage adds to what earlier attempts spent. run_worldmm.sh puts it into the
stage's runs.jsonl line. No model is loaded and nothing is patched for the Qwen arms beyond the class attribute.
Env: WORLDMM_OPENAI_CACHE (folder of WorldMM's reply cache; default its own .cache in the working directory),
WORLDMM_OPENAI_SERVICE_TIER (e.g. flex: half price, slower; default the API's), OPENAI_API_KEY (read by
WorldMM's OpenAIModel from the environment; never passed on a command line or written anywhere).
"""
import inspect
import atexit
import json
import os
import threading

from worldmm.llm import openai_gpt

PATH = os.environ.get("WORLDMM_USAGE_FILE")
FIELDS = ("calls", "input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens")
_stage, _lock = ["other"], threading.Lock()
usage = json.load(open(PATH)) if PATH and os.path.exists(PATH) else {}


class EmptyReply(RuntimeError):
    pass


def stage(name):
    """Name the pipeline step whose calls follow (captions, multiscale, triples, semantic, openie)."""
    _stage[0] = name


def save():
    if not PATH or not usage:   # the Qwen arms make no OpenAI call and get no file
        return
    with _lock:
        os.makedirs(os.path.dirname(os.path.abspath(PATH)), exist_ok=True)
        json.dump(usage, open(PATH + ".tmp", "w"), indent=1)
        os.replace(PATH + ".tmp", PATH)


def _count(model, resp):
    u = getattr(resp, "usage", None)
    got = {"calls": 1, "input_tokens": getattr(u, "input_tokens", 0) or 0,
           "cached_tokens": getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0,
           "output_tokens": getattr(u, "output_tokens", 0) or 0,
           "reasoning_tokens": getattr(getattr(u, "output_tokens_details", None), "reasoning_tokens", 0) or 0}
    with _lock:
        row = usage.setdefault(getattr(model, "usage_role", None) or _stage[0], {}).setdefault(
            str(getattr(resp, "model", None) or model.model_name), dict.fromkeys(FIELDS, 0))
        for k in FIELDS:
            row[k] += got[k]
        row["service_tier"] = getattr(resp, "service_tier", None)
        n = sum(r["calls"] for s in usage.values() for r in s.values())
    if n % 25 == 0:
        save()
    if not (getattr(resp, "output_text", None) or "").strip() and getattr(resp, "output_parsed", None) is None:
        raise EmptyReply(f"{model.model_name} returned an empty reply (status {getattr(resp, 'status', None)})")
    return resp


def _wrap(responses, model):
    for name in ("create", "parse"):
        call = getattr(responses, name)

        def counted(*a, _call=call, **kw):
            r = _call(*a, **kw)
            if inspect.isawaitable(r):   # the async client (its methods are not always coroutine functions themselves)
                async def later():
                    return _count(model, await r)
                return later()
            return _count(model, r)
        setattr(responses, name, counted)


_init = openai_gpt.OpenAIModel.__init__


def _counted_init(self, *a, **kw):
    if os.environ.get("WORLDMM_OPENAI_CACHE"):
        kw.setdefault("cache_dir", os.environ["WORLDMM_OPENAI_CACHE"])
    if os.environ.get("WORLDMM_OPENAI_SERVICE_TIER"):
        kw.setdefault("service_tier", os.environ["WORLDMM_OPENAI_SERVICE_TIER"])   # reaches every Responses call
    _init(self, *a, **kw)
    os.makedirs(os.path.dirname(os.path.abspath(self.cache_file_name)), exist_ok=True)
    _wrap(self.sync_client.responses, self)
    _wrap(self.async_client.responses, self)


openai_gpt.OpenAIModel.__init__ = _counted_init
atexit.register(save)
