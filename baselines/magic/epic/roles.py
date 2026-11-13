#!/usr/bin/env python3
"""Which endpoint answers which LLM role, for the arms that run MAGIC-Video's published configuration.

The local arms (m8, m27) send every LLM call to one server (MAGIC_LLM_BASE_URL). The reported arms
(run_magic.sh sets MAGIC_ROLES=1) split the calls by role, as the paper and upstream's defaults do:

  caption  the 30 s clip captions (a VLM)                       default qwen/qwen3.5-flash-02-23 (OpenRouter)
  memory   summaries, OpenIE, semantic memory, narrative chains  default openai/gpt-oss-120b      (OpenRouter)
  reason   the final answer (it sees the retrieved frames) and the time parser   default qwen/qwen3.5-flash-02-23
  control  the search/answer loop of each question (the paper's "online retrieval controller"), at temperature 0:
           the memory role's model and endpoint. MAGIC_CONTROLLER=reason gives it to the reason role with the
           responder's sampling instead, which is what the released code does (eval_egolife.py).

Per role R in CAPTION, MEMORY, REASON (all optional):
  MAGIC_<R>_MODEL     the model id sent to the endpoint
  MAGIC_<R>_BASE_URL  an OpenAI-compatible endpoint (default https://openrouter.ai/api/v1)
  MAGIC_<R>_KEY_ENV   the NAME of the environment variable that holds the endpoint's key (default
                      OPENROUTER_API_KEY; empty = no key: our own server, e.g. epic/hf_server.py or vLLM, and the
                      default for a 127.0.0.1 / localhost endpoint)
Keys are read from the environment at call time only: never from argv or a file, never logged or recorded.

Every call is counted per role (calls, prompt / completion / reasoning tokens, the cost when the endpoint reports
it, failed calls) in MAGIC_USAGE_FILE under MAGIC_STAGE, summed over reruns; run_magic.sh copies it into runs.jsonl.

    python roles.py table                # {role: {model, endpoint, key}} as JSON (key: the variable's name, or null)
    python roles.py check memory reason  # exit 1, with a message, when one of these roles' keys is not set
"""
import atexit
import json
import os
import sys
import threading
from urllib.parse import urlparse

ROUTED = os.environ.get("MAGIC_ROLES") == "1"
OPENROUTER = "https://openrouter.ai/api/v1"
DEFAULT = {"caption": "qwen/qwen3.5-flash-02-23", "memory": "openai/gpt-oss-120b", "reason": "qwen/qwen3.5-flash-02-23"}
COUNTS = ("calls", "prompt_tokens", "completion_tokens", "reasoning_tokens", "cost", "failed")
LOCK = threading.Lock()
USAGE = {}      # role -> counts of this process
FATAL = []      # a message once a call failed for a reason no retry fixes (no key, key refused, no credit)


def controller():
    c = os.environ.get("MAGIC_CONTROLLER") or "memory"
    if c not in ("memory", "reason"):
        sys.exit("MAGIC_CONTROLLER must be memory (the paper) or reason (the released code)")
    return c


def endpoint(role):
    """(model id, base URL, name of the key's environment variable or '')."""
    r = (controller() if role == "control" else role).upper()
    url = os.environ.get(f"MAGIC_{r}_BASE_URL") or OPENROUTER
    own = urlparse(url).hostname in ("127.0.0.1", "localhost")
    return (os.environ.get(f"MAGIC_{r}_MODEL") or DEFAULT[role], url,
            os.environ.get(f"MAGIC_{r}_KEY_ENV", "" if own else "OPENROUTER_API_KEY"))


def table():
    return {r: dict(zip(("model", "endpoint", "key"), (m, u, k or None))) for r in [*DEFAULT, "control"] for m, u, k in [endpoint(r)]}


def missing(roles):
    return [f"{r} ({m} at {u}): {k} is not set" for r in roles for m, u, k in [endpoint(r)] if k and not os.environ.get(k)]


def errors():
    """Failed calls of this process so far: a stage checks it before it keeps what it made."""
    with LOCK:
        return sum(u["failed"] for u in USAGE.values())


def _save():
    f, stage = os.environ.get("MAGIC_USAGE_FILE"), os.environ.get("MAGIC_STAGE", "")
    if not f or not USAGE:
        return
    with LOCK:
        allu = dict(BASE)
        old = BASE.get(stage, {})
        allu[stage] = {r: {**u, **{c: u[c] + old.get(r, {}).get(c, 0) for c in COUNTS},
                           "returned": sorted(set(u["returned"]) | set(old.get(r, {}).get("returned", [])))} for r, u in USAGE.items()}
        for r, u in BASE.get(stage, {}).items():   # roles only earlier runs of the stage called
            allu[stage].setdefault(r, u)
        with open(f + ".tmp", "w") as fh:
            json.dump(allu, fh, indent=1)
        os.replace(f + ".tmp", f)


try:    # earlier runs of this stage (a rerun adds to them)
    BASE = json.load(open(os.environ.get("MAGIC_USAGE_FILE", "")))
except (OSError, ValueError):
    BASE = {}
atexit.register(_save)


class Client:
    """Stands in for the OpenAI client inside WorldMM's OpenRouterModel (it calls client.chat.completions.create):
    sends the call to the role's endpoint and counts it."""

    def __init__(self, role):
        self.role = role
        self.model, self.url, self.key_env = endpoint(role)
        self.chat = self.completions = self
        self._api = None

    def create(self, model=None, messages=None, **kw):   # `model` is WorldMM's file tag, not an id: ignored
        from openai import APIStatusError, OpenAI
        with LOCK:
            u = USAGE.setdefault(self.role, {"model": self.model, "returned": [], **{c: 0 for c in COUNTS}})
        try:
            if FATAL:
                raise RuntimeError(FATAL[0])
            if self._api is None:
                key = os.environ.get(self.key_env) if self.key_env else "EMPTY"
                if not key:
                    FATAL.append(f"{self.key_env} is not set: role {self.role} ({self.model} at {self.url}) is a paid "
                                 "API, so this step runs only where that key is (job board: pool max)")
                    raise RuntimeError(FATAL[0])
                self._api = OpenAI(api_key=key, base_url=self.url, timeout=600 if self.key_env else 1800, max_retries=5)
            if not self.key_env:   # our own server: called as the local arms call it (non-thinking, seeded, bounded)
                kw.setdefault("extra_body", {}).setdefault("chat_template_kwargs", {"enable_thinking": False})
                kw.setdefault("seed", int(os.environ.get("MAGIC_SEED", "0")))
                kw.setdefault("max_tokens", int(os.environ.get("MAGIC_MAX_TOKENS", "4096")))
            res = self._api.chat.completions.create(model=self.model, messages=messages, **kw)
        except Exception as e:
            if isinstance(e, APIStatusError) and e.status_code in (401, 402, 403) and not FATAL:
                FATAL.append(f"role {self.role} ({self.model} at {self.url}) refused the key or has no credit "
                             f"(HTTP {e.status_code}): stopping")
            with LOCK:
                u["failed"] += 1
            raise
        us = getattr(res, "usage", None)
        with LOCK:
            u["calls"] += 1
            if getattr(res, "model", None) and res.model not in u["returned"]:   # the id the endpoint says answered
                u["returned"].append(res.model)
            if us is not None:
                u["prompt_tokens"] += us.prompt_tokens or 0
                u["completion_tokens"] += us.completion_tokens or 0
                u["reasoning_tokens"] += getattr(getattr(us, "completion_tokens_details", None), "reasoning_tokens", None) or 0
                u["cost"] += (getattr(us, "model_extra", None) or {}).get("cost") or 0   # OpenRouter reports dollars
            n = sum(x["calls"] for x in USAGE.values())
        if n % 25 == 0:
            _save()
        return res


def model(role, tag, **kwargs):
    """WorldMM's LLM object for a role: upstream's OpenRouterModel (its message handling, sampling arguments and
    JSON parsing, unchanged) with the role's client; `tag` is the name WorldMM puts in its file names. The
    submodule is not patched for this."""
    from worldmm.llm.openrouter import OpenRouterModel

    class RoleModel(OpenRouterModel):
        def __init__(self):   # not OpenRouterModel.__init__: that one needs OPENROUTER_API_KEY when it is built
            self.client, self.model_name, self.max_retries, self.kwargs = Client(role), tag, 3, kwargs
    return RoleModel()


def responder(tag, loop_prompt, **sampling):
    """WorldMM's respond model, split as the paper describes it: upstream's answer loop sends both its
    search-or-answer decisions and the final answer to one respond model, so this one object passes a call that
    starts with the loop's prompt (loop_prompt, the rendered memory_reasoning_unified template) to the control
    role, and every other call (the final answer) to the reason role with upstream's responder sampling.
    With MAGIC_CONTROLLER=reason both go to the reason role with that sampling, as in the released code."""
    answer = model("reason", tag, **sampling)
    if controller() == "reason":
        return answer
    control = model("control", tag, temperature=0.0)

    class Split:
        model_name = tag

        def generate(self, prompt, **kw):
            if isinstance(prompt, list) and prompt[:len(loop_prompt)] == loop_prompt:
                return control.generate(prompt, **kw)
            return answer.generate(prompt, **kw)
    return Split()


if __name__ == "__main__":
    if sys.argv[1:2] == ["table"]:
        print(json.dumps(table()))
    elif sys.argv[1:2] == ["check"]:
        bad = missing(sys.argv[2:])
        if bad:
            sys.exit("paid API role without its key in the environment (run this stage where the key is; job board: "
                     "pool max):\n  " + "\n  ".join(bad))
    else:
        sys.exit(__doc__)
