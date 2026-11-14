"""The official arms' OpenAI calls, counted (import after wmm_paths in every epic/ script that may make them).

WorldMM's OpenAIModel returns only the reply text. Here every OpenAIModel gets its Responses calls wrapped, so
each real API call (not a hit of WorldMM's sqlite reply cache) adds the tokens the API billed and the model id it
answered with to WORLDMM_USAGE_FILE, under the pipeline step set with stage() or the model's `usage_role`:

    {"captions": {"gpt-5-mini-2025-08-07": {"calls": 163, "input_tokens": ..., "cached_tokens": ...,
                                            "output_tokens": ..., "reasoning_tokens": ..., "service_tier": "default",
                                            "endpoint": "https://api.openai.com/v1/"}}, ...}

The gateway: with OPENROUTER_API_KEY set and OPENAI_API_KEY not, the same Responses calls go to OpenRouter
(OPENROUTER_BASE_URL, default https://openrouter.ai/api/v1) as openai/gpt-5-mini / openai/gpt-5, pinned to the
provider OpenAI with no fallback (OpenRouter also lists Azure for these models). The request is otherwise the
same, except that each image part says detail "auto": OpenRouter's schema requires the field, and auto is what
the OpenAI API assumes without it. The rows are then keyed by the id OpenRouter returns, with its endpoint and
"cost" (dollars OpenRouter charged, summed), and "provider" when the reply names who served it (a Responses reply
does not, unless asked with the header X-OpenRouter-Metadata, which is not sent).
WorldMM's reply cache keys on the model id, so replies of one gateway are never served as the other's.

An empty reply (no text and nothing parsed) is counted, since it was billed, and then raised as an error: WorldMM's
retry asks again, and its reply cache never stores it (a cached empty reply would be served on every rerun).
The file is cumulative: a resumed stage adds to what earlier attempts spent. run_worldmm.sh puts it into the
stage's runs.jsonl line. No model is loaded and nothing is patched for the Qwen arms beyond the class attribute.
Env: WORLDMM_OPENAI_CACHE (folder of WorldMM's reply cache; default its own .cache in the working directory),
WORLDMM_OPENAI_SERVICE_TIER (e.g. flex: half price, slower; default the API's), OPENAI_API_KEY or
OPENROUTER_API_KEY (read from the environment; never passed on a command line or written anywhere).
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
# the gateway (run_worldmm.sh keeps one key only): OpenRouter's key without OpenAI's sends the calls to OpenRouter
OPENROUTER = bool(os.environ.get("OPENROUTER_API_KEY")) and not os.environ.get("OPENAI_API_KEY")
OPENROUTER_URL = os.environ.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
PROVIDER = {"only": ["openai"], "allow_fallbacks": False}   # OpenAI itself answers, or the call fails


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


def _count(model, resp, endpoint):
    u = getattr(resp, "usage", None)
    provider = (getattr(resp, "model_extra", None) or {}).get("provider")   # OpenRouter reports this and the cost
    cost = (getattr(u, "model_extra", None) or {}).get("cost")
    got = {"calls": 1, "input_tokens": getattr(u, "input_tokens", 0) or 0,
           "cached_tokens": getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0,
           "output_tokens": getattr(u, "output_tokens", 0) or 0,
           "reasoning_tokens": getattr(getattr(u, "output_tokens_details", None), "reasoning_tokens", 0) or 0}
    with _lock:
        row = usage.setdefault(getattr(model, "usage_role", None) or _stage[0], {}).setdefault(
            str(getattr(resp, "model", None) or model.model_name), dict.fromkeys(FIELDS, 0))
        for k in FIELDS:
            row[k] += got[k]
        row.update(service_tier=getattr(resp, "service_tier", None), endpoint=endpoint)
        if provider:
            row["provider"] = provider
        if cost is not None:
            row["cost"] = round(row.get("cost", 0) + cost, 6)
        n = sum(r["calls"] for s in usage.values() for r in s.values())
    if n % 25 == 0:
        save()
    if not (getattr(resp, "output_text", None) or "").strip() and getattr(resp, "output_parsed", None) is None:
        raise EmptyReply(f"{model.model_name} returned an empty reply (status {getattr(resp, 'status', None)})")
    return resp


def _auto_detail(messages):
    """The same input with detail "auto" on every image part that names none (copies; the caller's stay as sent)."""
    part = lambda p: {"detail": "auto", **p} if isinstance(p, dict) and p.get("type") == "input_image" else p
    return [{**m, "content": [part(p) for p in m["content"]]} if isinstance(m, dict) and isinstance(m.get("content"), list)
            else m for m in messages] if isinstance(messages, list) else messages


def _wrap(client, model):
    responses, endpoint = client.responses, str(client.base_url)
    for name in ("create", "parse"):
        call = getattr(responses, name)

        def counted(*a, _call=call, **kw):
            if OPENROUTER:
                kw["extra_body"] = {**(kw.get("extra_body") or {}), "provider": PROVIDER}
                kw["input"] = _auto_detail(kw.get("input"))
            r = _call(*a, **kw)
            if inspect.isawaitable(r):   # the async client (its methods are not always coroutine functions themselves)
                async def later():
                    return _count(model, await r, endpoint)
                return later()
            return _count(model, r, endpoint)
        setattr(responses, name, counted)


_init = openai_gpt.OpenAIModel.__init__


def _counted_init(self, *a, **kw):
    if os.environ.get("WORLDMM_OPENAI_CACHE"):
        kw.setdefault("cache_dir", os.environ["WORLDMM_OPENAI_CACHE"])
    if os.environ.get("WORLDMM_OPENAI_SERVICE_TIER"):
        kw.setdefault("service_tier", os.environ["WORLDMM_OPENAI_SERVICE_TIER"])   # reaches every Responses call
    if OPENROUTER:
        kw.setdefault("api_key", os.environ["OPENROUTER_API_KEY"])
    _init(self, *a, **kw)
    if OPENROUTER:   # gpt-5-mini -> openai/gpt-5-mini (OpenRouter's id of the snapshot WorldMM pins; also the cache key)
        self.model_name = "openai/" + next(k for k, v in openai_gpt.MODEL_DICT.items() if v == self.model_name)
        self.sync_client.base_url = self.async_client.base_url = OPENROUTER_URL
    os.makedirs(os.path.dirname(os.path.abspath(self.cache_file_name)), exist_ok=True)
    _wrap(self.sync_client, self)
    _wrap(self.async_client, self)


openai_gpt.OpenAIModel.__init__ = _counted_init
atexit.register(save)
