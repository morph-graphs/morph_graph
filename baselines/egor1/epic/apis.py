"""The model of each Ego-R1 step, and the two clients this port adds for the steps upstream has no code for
(the memory build) or only inside its benchmark script (the final answer).

The models are the paper's, with one replacement: Gemini 1.5 Pro is retired, so the two steps that used it (30 s
captions, video_llm) use the Gemini of the arm (ARMS, use_arm). Arm gemini25pro (Anubhav, 2026-11-13): Gemini
2.5 Pro, which Google no longer serves to a key that had not used it (404 on ours, 2026-11-14) and lists for
shutdown on 2026-12-01, after which requests are routed to another model: from that day its Gemini steps refuse
to run (need). Arm gemini36flash (2026-11-14): Gemini 3.6 Flash; the decision named 3.5 Flash, whose endpoint
Google routes to 3.6 Flash. A reply from another model than the one asked is an error on every arm (track).
Arm gemini36flash-gpt41 (Anubhav, 2026-11-14) is gemini36flash with GPT-4.1 in place of GPT-4o in the rag and
vlm tools: the machines that run the row have no GPT-4o. Its memory is gemini36flash's (the same models make it).

EGOR1_GEMINI_INLINE=1 (opt-in, for an endpoint without Google's Files API) sends each video inside the request
instead of uploading it first (track). Not tried against any endpoint.

Keys come from the environment only, never from a command line or a file of the repo:
  GEMINI_API_KEY                                   Gemini
  AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY     Azure OpenAI with deployments named gpt-4o and gpt-4.1 (upstream's
                                                   way), or
  OPENAI_API_KEY                                   a plain OpenAI key (needs ../epic-morph.patch on upstream), or
  OPENROUTER_API_KEY                               without the ones above: the same GPT calls through OpenRouter
                                                   (OPENROUTER_BASE_URL, default https://openrouter.ai/api/v1) as
                                                   openai/gpt-4o and openai/gpt-4.1, pinned to the provider OpenAI
                                                   with no fallback (track). Gemini never goes through it.
A step whose key is missing stops with exit code 5 before doing anything."""
import datetime
import json
import os
import sys
import time

MODELS = {   # role -> model; in the paper / official repo (commit 69177ee) both Gemini roles are gemini-1.5-pro
    "agent": "Ego-R1/Ego-R1-Agent-3B",   # open weights, run locally (Qwen2.5-3B-Instruct, SFT + GRPO)
    "caption": "gemini-2.5-pro",         # 30 s clip -> dense caption with dialogue (paper, H-RAG construction)
    "summary": "gpt-4.1",                # captions -> 10 min -> 1 hour -> 1 day summaries (paper)
    "rag": "gpt-4o",                     # keyword retrieval over the logs (r1rag/utils.py call_gpt4; EGOR1_RAG_MODEL, our patch)
    "video_llm": "gemini-2.5-pro",       # egor1_videollm_gemini/video_llm.py (model from EGOR1_GEMINI_MODEL, our patch)
    "vlm": "gpt-4o",                     # egor1_vlm/utils.py GPT (EGOR1_VLM_MODEL, our patch)
    "answer": "gpt-4.1",                 # Ego-R1-Agent/eval/infer_bench_summ.py summ_client
}
DEFAULT = dict(MODELS)
ARMS = {   # arm -> the roles whose model differs from DEFAULT
    "gemini25pro": {},
    "gemini36flash": {"caption": "gemini-3.6-flash", "video_llm": "gemini-3.6-flash"},
    "gemini36flash-gpt41": {"caption": "gemini-3.6-flash", "video_llm": "gemini-3.6-flash", "rag": "gpt-4.1", "vlm": "gpt-4.1"},
}
SHUTDOWN = {"gemini-2.5-pro": datetime.date(2026, 12, 1)}   # Google's deprecation page, read 2026-11-13 (3.6 Flash: none announced, 2026-11-14)
USAGE = None   # path of a jsonl: one line per API call of this process (track), for the real cost of a run
OPENROUTER_URL = os.environ.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
PROVIDER = {"only": ["openai"], "allow_fallbacks": False}   # OpenAI itself answers, or the call fails


def use_arm(arm):
    """Set the models of the arm's steps (arm llava keeps the defaults for all but its local video_llm)."""
    MODELS.update(DEFAULT, **ARMS.get(arm, {}))


def openai_keys():
    if os.environ.get("AZURE_OPENAI_ENDPOINT") and os.environ.get("AZURE_OPENAI_API_KEY"):
        return "azure"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "openrouter" if os.environ.get("OPENROUTER_API_KEY") else None


def need(*roles):
    """Stop (exit 5) unless the keys of every closed model these roles use are set (and Gemini 2.5 Pro still exists)."""
    missing = []
    for r in roles:
        m = MODELS[r]
        if m.startswith("gemini"):
            if not os.environ.get("GEMINI_API_KEY"):
                missing.append(f"{r} = {m}: GEMINI_API_KEY is not set")
            if m in SHUTDOWN and datetime.date.today() >= SHUTDOWN[m]:
                missing.append(f"{r} = {m}: shut down on {SHUTDOWN[m]} (requests go to another model); choose a model first")
        elif m.startswith("gpt") and not openai_keys():
            missing.append(f"{r} = {m}: set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY, or OPENAI_API_KEY, "
                           "or OPENROUTER_API_KEY (through OpenRouter)")
    if missing:
        for x in missing:
            print(f"needs a paid API: {x}", file=sys.stderr)
        sys.exit(5)


ROLE = None    # the step making the current call (caption, summary, rag, video_llm, vlm, answer)
QUESTION = None   # qa: the key of the question the current call is made for


def _log(asked, got, tin, tout, think=0, **more):
    if USAGE:
        with open(USAGE, "a") as f:
            f.write(json.dumps({"role": ROLE, "asked": asked, "model": got, "in": tin, "out": tout, "thinking": think,
                                "t": int(time.time()), "run": os.environ.get("EGOR1_RUN"), "q": QUESTION, **more}) + "\n")


class Inline:
    """Stands for an uploaded file when the video goes inside the request instead (EGOR1_GEMINI_INLINE): what
    upstream's video_llm and gemini_video read of a file (its state, name and uri)."""

    def __init__(self, path):
        from google.genai import types
        self.path, self.name, self.uri, self.state = path, "inline:" + path, path, types.FileState.ACTIVE


class WrongModel(SystemExit):
    """A Gemini reply came from another model than the one asked: stop the whole run (exit code 6). Not an
    Exception, so neither upstream's handlers nor the tools' retry logic take it for a failed call."""


def track():
    """Log every chat completion and Gemini generation of this process, upstream's included (their clients are
    built inside upstream's functions): the model that answered and the tokens, thinking counted apart. A Gemini
    reply from a model other than the one asked is an error: Google routes a retired id to another model
    (gemini-3.5-flash to 3.6 Flash today, gemini-2.5-pro after its shutdown date) (WrongModel). Wraps the two SDK methods once. A request is changed only when its client
    points at OpenRouter (apis.gpt41, and upstream's clients through our patch): the model becomes openai/<model>
    and the provider pin is added; the line then has the provider that served it and the dollars charged.
    With EGOR1_GEMINI_INLINE set, a Files upload returns an Inline instead of uploading, and generate_content
    sends that file's bytes as an inline video part (delete then has nothing to remove)."""
    from google.genai import files as gf, models as gm, types
    from openai.resources.chat.completions import Completions
    if getattr(Completions.create, "egor1", False):
        return
    chat, gen, up, rm = Completions.create, gm.Models.generate_content, gf.Files.upload, gf.Files.delete

    def upload(self, *a, **k):
        return Inline(str(k["file"])) if os.environ.get("EGOR1_GEMINI_INLINE") else up(self, *a, **k)

    def delete(self, *a, **k):
        return None if str(k.get("name")).startswith("inline:") else rm(self, *a, **k)

    def create(self, *a, **k):
        endpoint = str(self._client.base_url)
        if endpoint.rstrip("/") == OPENROUTER_URL.rstrip("/"):
            k["model"] = "openai/" + k["model"]
            k["extra_body"] = {**(k.get("extra_body") or {}), "provider": PROVIDER}
        r = chat(self, *a, **k)
        u = getattr(r, "usage", None)
        _log(k.get("model"), getattr(r, "model", None), u.prompt_tokens if u else 0, u.completion_tokens if u else 0,
             cached=getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0, endpoint=endpoint,
             provider=(getattr(r, "model_extra", None) or {}).get("provider"),      # OpenRouter reports both
             cost=(getattr(u, "model_extra", None) or {}).get("cost"))
        return r

    def generate_content(self, *a, **k):
        if isinstance(k.get("contents"), list):
            k["contents"] = [types.Part.from_bytes(data=open(c.path, "rb").read(), mime_type="video/mp4")
                             if isinstance(c, Inline) else c for c in k["contents"]]
        r = gen(self, *a, **k)
        u, asked, got = r.usage_metadata, k.get("model") or "", r.model_version or ""
        _log(asked, got, *((u.prompt_token_count or 0, u.candidates_token_count or 0, u.thoughts_token_count or 0) if u else (0, 0, 0)))
        if not got.removeprefix("models/").startswith(asked.removeprefix("models/")):
            print(f"asked {asked}, answered by {got!r}: stopping (is {asked} shut down?)", file=sys.stderr)
            raise WrongModel(6)
        return r
    create.egor1 = True
    Completions.create, gm.Models.generate_content, gf.Files.upload, gf.Files.delete = create, generate_content, upload, delete


def usage_summary(path, run=None):
    """{"<role> <model that answered>": {"calls", "in", "out", "thinking"}} of the log's lines (of one stage's
    run id, EGOR1_RUN, when given: jobs of one video can share the log). Chat completions also have "cached"
    (input tokens billed as cached) and "endpoint"; through OpenRouter "provider" (who served the calls) and
    "cost" (dollars charged, summed)."""
    out = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                x = json.loads(line)
                if run and x.get("run") != run:
                    continue
                d = out.setdefault(f"{x['role']} {x['model']}", {"calls": 0, "in": 0, "out": 0, "thinking": 0})
                d["calls"] += 1
                for k in ("in", "out", "thinking"):
                    d[k] += x.get(k) or 0
                if x.get("endpoint"):
                    d["cached"] = d.get("cached", 0) + (x.get("cached") or 0)
                    d["endpoint"] = x["endpoint"]
                if x.get("provider"):
                    d["provider"] = x["provider"]
                if x.get("cost") is not None:
                    d["cost"] = round(d.get("cost", 0) + x["cost"], 6)
    return out


def gpt41(role, messages, max_tokens, temperature=1.0, top_p=1.0):
    """One GPT-4.1 chat completion (as infer_bench_summ.py's summ_client; Azure, a plain OpenAI key, or OpenRouter)."""
    keys = openai_keys()
    if keys == "azure":
        from openai import AzureOpenAI
        client = AzureOpenAI(azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"], api_key=os.environ["AZURE_OPENAI_API_KEY"],
                             api_version="2025-01-01-preview")
    elif keys == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    else:      # track() names the model openai/gpt-4.1 and pins the provider
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url=OPENROUTER_URL)
    global ROLE
    ROLE = role
    track()
    r = client.chat.completions.create(model=MODELS[role], messages=messages, max_tokens=max_tokens, temperature=temperature,
                                       top_p=top_p, frequency_penalty=0, presence_penalty=0, stop=None, stream=False)
    return r.choices[0].message.content or ""


def gemini_video(role, path, prompt):
    """Upload one clip and ask Gemini about it (the same calls as egor1_videollm_gemini/video_llm.py)."""
    from google import genai
    global ROLE
    ROLE = role
    track()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    f = client.files.upload(file=path)
    t0 = time.time()
    while f.state.name == "PROCESSING":
        if time.time() - t0 > 300:
            raise TimeoutError(f"Gemini is still processing {path}")
        time.sleep(5)
        f = client.files.get(name=f.name)
    if f.state.name != "ACTIVE":
        raise RuntimeError(f"Gemini file {f.name} is {f.state.name}")
    try:
        r = client.models.generate_content(model=MODELS[role], contents=[f, prompt])
    finally:
        client.files.delete(name=f.name)
    return r.text or ""
