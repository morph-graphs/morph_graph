"""The model of each Ego-R1 step, and the two clients this port adds for the steps upstream has no code for
(the memory build) or only inside its benchmark script (the final answer).

The models are the paper's, with one replacement decided by Anubhav (2026-11-13): Gemini 1.5 Pro is retired, so
the two steps that used it (30 s captions, video_llm) use Gemini 2.5 Pro. Google's deprecation page lists
gemini-2.5-pro for shutdown on 2026-12-01, after which requests are routed to another model: from that day the
Gemini steps refuse to run (need), and a reply from another model is an error (gemini_video).

Keys come from the environment only, never from a command line or a file of the repo:
  GEMINI_API_KEY                                   Gemini
  AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY     Azure OpenAI with deployments named gpt-4o and gpt-4.1 (upstream's
                                                   way), or
  OPENAI_API_KEY                                   a plain OpenAI key (needs ../epic-morph.patch on upstream)
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
    "rag": "gpt-4o",                     # keyword retrieval over the logs (r1rag/utils.py call_gpt4)
    "video_llm": "gemini-2.5-pro",       # egor1_videollm_gemini/video_llm.py (model from EGOR1_GEMINI_MODEL, our patch)
    "vlm": "gpt-4o",                     # egor1_vlm/utils.py GPT
    "answer": "gpt-4.1",                 # Ego-R1-Agent/eval/infer_bench_summ.py summ_client
}
GEMINI_SHUTDOWN = datetime.date(2026, 12, 1)   # gemini-2.5-pro, Google's deprecation page, read 2026-11-13
USAGE = None   # path of a jsonl; one line per call made here (tokens), for the real cost of a run


def openai_keys():
    if os.environ.get("AZURE_OPENAI_ENDPOINT") and os.environ.get("AZURE_OPENAI_API_KEY"):
        return "azure"
    return "openai" if os.environ.get("OPENAI_API_KEY") else None


def need(*roles):
    """Stop (exit 5) unless the keys of every closed model these roles use are set (and Gemini 2.5 Pro still exists)."""
    missing = []
    for r in roles:
        m = MODELS[r]
        if m.startswith("gemini"):
            if not os.environ.get("GEMINI_API_KEY"):
                missing.append(f"{r} = {m}: GEMINI_API_KEY is not set")
            if datetime.date.today() >= GEMINI_SHUTDOWN:
                missing.append(f"{r} = {m}: shut down on {GEMINI_SHUTDOWN} (requests go to another model); choose a model first")
        elif m.startswith("gpt") and not openai_keys():
            missing.append(f"{r} = {m}: set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY, or OPENAI_API_KEY")
    if missing:
        for x in missing:
            print(f"needs a paid API: {x}", file=sys.stderr)
        sys.exit(5)


def _log(role, model, tin, tout):
    if USAGE:
        with open(USAGE, "a") as f:
            f.write(json.dumps({"role": role, "model": model, "in": tin, "out": tout, "t": int(time.time())}) + "\n")


def gpt41(role, messages, max_tokens, temperature=1.0, top_p=1.0):
    """One GPT-4.1 chat completion (as infer_bench_summ.py's summ_client; Azure, or a plain OpenAI key)."""
    if openai_keys() == "azure":
        from openai import AzureOpenAI
        client = AzureOpenAI(azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"], api_key=os.environ["AZURE_OPENAI_API_KEY"],
                             api_version="2025-01-01-preview")
    else:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    r = client.chat.completions.create(model=MODELS[role], messages=messages, max_tokens=max_tokens, temperature=temperature,
                                       top_p=top_p, frequency_penalty=0, presence_penalty=0, stop=None, stream=False)
    _log(role, MODELS[role], r.usage.prompt_tokens, r.usage.completion_tokens)
    return r.choices[0].message.content or ""


def gemini_video(role, path, prompt):
    """Upload one clip and ask Gemini about it (the same calls as egor1_videollm_gemini/video_llm.py)."""
    from google import genai
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
    if not (r.model_version or "").startswith(MODELS[role]):
        raise RuntimeError(f"asked {MODELS[role]}, answered by {r.model_version}")
    u = r.usage_metadata
    _log(role, MODELS[role], u.prompt_token_count, (u.candidates_token_count or 0) + (u.thoughts_token_count or 0))
    return r.text or ""
