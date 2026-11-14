#!/usr/bin/env python3
"""The g5m arms' OpenAI calls, counted.

MERIT's OpenAIModel returns only the reply text. Importing this module wraps every OpenAIModel's Chat Completions
calls, so each real API call (not a hit of MERIT's sqlite reply cache) appends one line to MERIT_USAGE_LOG: the
pipeline role, the model id the API answered with, the service tier and the tokens it billed. run_merit.sh sums
the lines into the stage's runs.jsonl entry (summary()):

    {"captions": {"gpt-5-mini-2025-08-07": {"calls": 35, "input_tokens": ..., "cached_tokens": ...,
                                            "output_tokens": ..., "reasoning_tokens": ..., "service_tier": "default",
                                            "endpoint": "https://api.openai.com/v1/"}},
     "keys": {...}}            (qa: "search", "filter", "answer")

The gateway: with OPENROUTER_API_KEY set and OPENAI_API_KEY not, the same calls go to OpenRouter instead
(OPENROUTER_BASE_URL, default https://openrouter.ai/api/v1) as openai/gpt-5-mini / openai/gpt-5, pinned to the
provider OpenAI with no fallback (OpenRouter also lists Azure for these models). Nothing else in the request
changes. The rows are then keyed by the id OpenRouter returns, with its endpoint, "provider" (who served the
call, as OpenRouter names it) and "cost" (dollars OpenRouter charged, summed). MERIT's reply cache keys on the
model id, so replies of one gateway are never served as the other's.

An empty reply (no text and nothing parsed) is counted, since it was billed, and then raised as an error: MERIT's
retry asks again, and its reply cache never stores it (a cached empty reply would be served on every rerun).

MERIT's own scripts are started through this file, so their worker processes import it too:

    python epic/openai_usage.py keys <preprocess/extract_keys.py arguments>
    python epic/openai_usage.py eval <eval/evaluate_egolife.py arguments>

Env: MERIT_USAGE_LOG (no counting without it), MERIT_OPENAI_CACHE (folder of MERIT's reply cache; default .cache in
the working directory), MERIT_SERVICE_TIER (e.g. flex), OPENAI_API_KEY or OPENROUTER_API_KEY (read from the
environment; never passed on a command line or written anywhere).
"""
import inspect
import json
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("preprocess", "eval", "src"):
    sys.path.insert(0, os.path.join(HERE, "..", "MERIT", d))
from merit.llm import openai_gpt  # noqa: E402

FIELDS = ("calls", "input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens")
# how each request starts (MERIT's prompts): the system prompt, or for the two memory calls the user turn
ROLES = (("You are a reasoning agent", "search"), ("You are a helpful assistant that extracts relevant information", "filter"),
         ("You are an AI assistant that answers questions", "answer"), ("You are extracting retrieval keys", "keys"))
_lock = threading.Lock()
# the gateway (run_merit.sh keeps one key only): OpenRouter's key without OpenAI's sends the calls to OpenRouter
OPENROUTER = bool(os.environ.get("OPENROUTER_API_KEY")) and not os.environ.get("OPENAI_API_KEY")
OPENROUTER_URL = os.environ.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
PROVIDER = {"only": ["openai"], "allow_fallbacks": False}   # OpenAI itself answers, or the call fails


class EmptyReply(RuntimeError):
    pass


def role(messages):
    c = messages[0].get("content") if messages else ""
    parts = isinstance(c, list)   # frames then the prompt: a caption, unless its text is one of the prompts above
    text = next((x.get("text") for x in c if isinstance(x, dict) and x.get("text")), "") if parts else (c or "")
    return next((r for start, r in ROLES if text.lstrip().startswith(start)), "captions" if parts else "other")


def _count(kw, resp, endpoint):
    path = os.environ.get("MERIT_USAGE_LOG")
    if path:
        u = getattr(resp, "usage", None)
        line = {"t": round(time.time(), 1), "role": role(kw.get("messages") or []), "requested": kw.get("model"),
                "model": getattr(resp, "model", None), "service_tier": getattr(resp, "service_tier", None),
                "endpoint": endpoint, "provider": (getattr(resp, "model_extra", None) or {}).get("provider"),
                "cost": (getattr(u, "model_extra", None) or {}).get("cost"),   # OpenRouter reports both
                "input_tokens": getattr(u, "prompt_tokens", 0) or 0,
                "cached_tokens": getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0,
                "output_tokens": getattr(u, "completion_tokens", 0) or 0,
                "reasoning_tokens": getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", 0) or 0}
        with _lock, open(path, "a") as f:   # a line per call: a killed stage loses nothing it paid for
            f.write(json.dumps(line) + "\n")
    msg = getattr((getattr(resp, "choices", None) or [None])[0], "message", None)
    if not (getattr(msg, "content", None) or "").strip() and getattr(msg, "parsed", None) is None:
        raise EmptyReply(f"{kw.get('model')} returned an empty reply")
    return resp


def _wrap(client):
    completions, endpoint = client.chat.completions, str(client.base_url)
    for name in ("create", "parse"):
        call = getattr(completions, name)

        def counted(*a, _call=call, **kw):
            if OPENROUTER:
                kw["extra_body"] = {**(kw.get("extra_body") or {}), "provider": PROVIDER}
            r = _call(*a, **kw)
            if inspect.isawaitable(r):   # the async client (its methods are not always coroutine functions themselves)
                async def later():
                    return _count(kw, await r, endpoint)
                return later()
            return _count(kw, r, endpoint)
        setattr(completions, name, counted)


_init = openai_gpt.OpenAIModel.__init__


def _counted_init(self, *a, **kw):
    if os.environ.get("MERIT_OPENAI_CACHE"):
        kw.setdefault("cache_dir", os.environ["MERIT_OPENAI_CACHE"])
    if os.environ.get("MERIT_SERVICE_TIER") and not kw.get("service_tier"):
        kw["service_tier"] = os.environ["MERIT_SERVICE_TIER"]   # flex also sets MERIT's 600 s request timeout
    if OPENROUTER:
        kw.setdefault("api_key", os.environ["OPENROUTER_API_KEY"])
    _init(self, *a, **kw)
    if OPENROUTER:   # gpt-5-mini -> openai/gpt-5-mini (OpenRouter's id of the snapshot MERIT pins; also the cache key)
        self.model_name = "openai/" + next(k for k, v in openai_gpt.MODEL_DICT.items() if v == self.model_name)
        self.sync_client.base_url = self.async_client.base_url = OPENROUTER_URL
    os.makedirs(os.path.dirname(os.path.abspath(self.cache_file_name)), exist_ok=True)
    _wrap(self.sync_client)
    _wrap(self.async_client)


openai_gpt.OpenAIModel.__init__ = _counted_init


def summary(path):
    """{role: {model id answered: summed FIELDS, service_tier, endpoint; through OpenRouter also provider and the
    summed cost}} of a usage log ({} when no call was made)."""
    out = {}
    for line in open(path) if os.path.exists(path) else []:
        r = json.loads(line)
        row = out.setdefault(r["role"], {}).setdefault(str(r["model"] or r["requested"]), dict.fromkeys(FIELDS, 0))
        row["calls"] += 1
        for k in FIELDS[1:]:
            row[k] += r[k]
        row.update(service_tier=r["service_tier"], endpoint=r.get("endpoint"))
        if r.get("provider"):
            row["provider"] = r["provider"]
        if r.get("cost") is not None:
            row["cost"] = round(row.get("cost", 0) + r["cost"], 6)
    return out


if __name__ == "__main__":
    what = sys.argv.pop(1)
    if what == "keys":
        from extract_keys import main
    elif what == "eval":
        from evaluate_egolife import main
    else:
        raise SystemExit("usage: openai_usage.py <keys|eval> <the MERIT script's arguments>")
    main()
