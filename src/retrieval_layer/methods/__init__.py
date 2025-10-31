"""
Retrieval methods for the memory system.
"""

# 避免导入需要 faiss 的模块
try:
    from .vector_search import VectorSearch
except ImportError:
    VectorSearch = None

from .keyword_search import KeywordSearch
# from .hybrid_search import HybridSearch

__all__ = [
    "KeywordSearch", 
]

if VectorSearch is not None:
    __all__.append("VectorSearch") 