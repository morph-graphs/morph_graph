from .gdb_networkx import NetworkXStorage
from .kv_json import JsonKVStorage
from .vdb_nanovectordb import NanoVectorDBStorage


def __getattr__(name: str):
    """Lazy-load optional storage backends with native/heavy deps."""
    if name == "HNSWVectorStorage":
        from .vdb_hnswlib import HNSWVectorStorage

        return HNSWVectorStorage
    if name == "Neo4jStorage":
        from .gdb_neo4j import Neo4jStorage

        return Neo4jStorage
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
