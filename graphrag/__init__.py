__version__ = "0.0.8.2"
__author__ = "Jianbai Ye"
__url__ = "https://github.com/gusye1234/nano-graphrag"

# dp stands for data pack


def __getattr__(name: str):
    """Lazy-load legacy nano-graphrag exports; avoids heavy imports on package init."""
    if name in ("GraphRAG", "QueryParam"):
        from .graphrag import GraphRAG, QueryParam

        return GraphRAG if name == "GraphRAG" else QueryParam
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
