"""The pinned stack imports (no GPU needed): run by setup_env.sh and the image build, with EgoLife/EgoGPT,
EgoLife/EgoRAG and epic/ on PYTHONPATH."""
import build_memory  # noqa: F401  swaps pysqlite3 in for sqlite3, as every script here does before Chroma
import chromadb, decord, deepspeed, librosa, open_clip, openai, torch, transformers, whisper  # noqa: E401,F401
from egogpt.model.builder import load_pretrained_model  # noqa: F401
from egorag.agents.RagAgent import RagAgent  # noqa: F401
from egorag.utils import gen_event  # noqa: F401

assert torch.__version__.startswith("2.1.2") and transformers.__version__ == "4.45.2", (torch.__version__, transformers.__version__)
assert chromadb.__version__ == "0.5.20" and openai.__version__ == "1.55.3", (chromadb.__version__, openai.__version__)
print("ok: torch", torch.__version__, "cuda", torch.version.cuda, "transformers", transformers.__version__,
      "chromadb", chromadb.__version__, "open_clip", open_clip.__version__, "openai", openai.__version__)
