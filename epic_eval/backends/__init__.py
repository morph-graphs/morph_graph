"""Model backends. Each module exposes the same four functions:

    make_client(mcfg) -> client
    open_context(client, mcfg, video_path | None, system_instruction, n_questions) -> ctx
    ask(client, mcfg, ctx, prompt, n_options) -> {"pred_letter", "raw_text", "usage", ...}
    close_context(client, ctx)

Add a local VLM by writing e.g. ``qwen3_vl.py`` with those functions (its
"context" can be decoded frames or a prefilled KV cache) and registering it here.
"""

import importlib

BACKENDS = {"gemini": "epic_eval.backends.gemini"}


def get_backend(name: str):
    return importlib.import_module(BACKENDS[name])
