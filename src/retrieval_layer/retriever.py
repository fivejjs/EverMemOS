"""
Unified retrieval interface for different search methods.
统一的检索接口，支持多种检索方法。
"""

import asyncio
import os
import logging
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod

# Try relative imports first, fallback to absolute imports for standalone usage
try:
    from .methods.vector_search import VectorSearch
except ImportError:
    # Fallback for standalone usage
    try:
        from methods.vector_search import VectorSearch
    except ImportError as e:
        # If imports still fail, create dummy classes/functions
        print(f"⚠️ Warning: Some retrieval methods not available: {e}")
            
        class VectorSearch:
            def __init__(self, *args, **kwargs): pass
            async def search(self, *args, **kwargs): return []

# 单独导入 KeywordSearch，避免 vector_search 的 faiss 依赖问题
try:
    from .methods.keyword_search import KeywordSearch
except ImportError:
    try:
        from methods.keyword_search import KeywordSearch
    except ImportError as e:
        print(f"⚠️ Warning: KeywordSearch not available: {e}")
        
        class KeywordSearch:
            def __init__(self, *args, **kwargs): pass
            async def search(self, *args, **kwargs): return []

logger = logging.getLogger(__name__)
# 

class BaseRetriever(ABC):
    """Abstract base class for retrievers."""
    
    @abstractmethod
    async def search(self, query: str, top_k: int = 10) -> List[str]:
        """Execute search and return results."""
        pass


class DynamicMemoryRetriever(BaseRetriever):
    """Dynamic Memory-based retriever using direct BM25 index files."""
    
    def __init__(self, version: str = "default", data_base: str = "memory_base"):
        """
        Initialize Dynamic Memory retriever.
        
        Args:
            version: Version identifier for the memory base
            data_base: Base path for memory storage
        """
        self.version = version
        self.data_base = data_base
        self._initialized = False
        self.available_users = set()
        self.index_dir = None
        self.nltk_available = False
        self.stop_words = set()
        self.stemmer = None
        
    async def initialize(self):
        """Initialize the retriever by checking available BM25 indices."""
        if self._initialized:
            return
            
        try:
            # Setup path to BM25 indices - ensure absolute path
            if Path(self.data_base).is_absolute():
                memory_base_path = Path(self.data_base) / "dynamic_memory_base" / "memory"
            else:
                # Convert relative path to absolute
                current_dir = Path(__file__).parent.parent.parent  # Go up to workspace root
                memory_base_path = current_dir / self.data_base / "dynamic_memory_base" / "memory"
            
            index_dir = memory_base_path / "bm25"
            self.index_dir = index_dir
            
            if not index_dir.exists():
                logger.warning(f"BM25 index directory not found: {index_dir}")
                logger.info("You may need to call build_bm25_idx() first on DynamicMemoryStorageV2")
                self.available_users = set()
                self._initialized = True
                return
            
            # Initialize NLTK components for query tokenization (same as index building)
            try:
                import nltk
                from nltk.corpus import stopwords
                from nltk.stem import PorterStemmer
                from nltk.tokenize import word_tokenize
                
                # Ensure NLTK data
                try:
                    nltk.data.find("tokenizers/punkt")
                    nltk.data.find("corpora/stopwords")
                    self.stop_words = set(stopwords.words("english"))
                    self.stemmer = PorterStemmer()
                    self.nltk_available = True
                    logger.debug("NLTK components initialized successfully")
                except:
                    logger.warning("NLTK data not available, using fallback tokenization")
                    self.stop_words = set()
                    self.stemmer = None
                    self.nltk_available = False
            except ImportError:
                logger.warning("NLTK not installed, using fallback tokenization")
                self.stop_words = set()
                self.stemmer = None
                self.nltk_available = False
            
            # Discover available users from indices
            for index_file in index_dir.glob("bm25_index_*.pkl"):
                user_id = index_file.stem.replace("bm25_index_", "")
                self.available_users.add(user_id)
            
            logger.info(f"Initialized BM25 retriever with users: {list(self.available_users)}")
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize Dynamic Memory retriever: {e}")
            raise
    
    async def search(self, query: str, top_k: int = 10, owner_id: str = "default") -> List[str]:
        """
        Search using direct BM25 index files.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            owner_id: User ID to search within
            
        Returns:
            List of relevant episode contents
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            if owner_id not in self.available_users:
                logger.warning(f"No BM25 index found for user: {owner_id}")
                logger.info(f"Available users: {list(self.available_users)}")
                return []
            
            # Load user's BM25 index
            import pickle
            index_file = self.index_dir / f"bm25_index_{owner_id}.pkl"
            
            with open(index_file, 'rb') as f:
                index_data = pickle.load(f)
            
            episodes = index_data["episodes"]
            bm25_index = index_data.get("bm25_index")
            
            if not episodes:
                return []
            
            # If no BM25 index exists, build one on-the-fly
            if not bm25_index:
                try:
                    from rank_bm25 import BM25Okapi
                    
                    # Extract content and build BM25 index
                    documents = []
                    for episode in episodes:
                        # Handle both dict and Episode object formats
                        if hasattr(episode, 'summary'):  # Nemori Episode object
                            content = episode.summary if episode.summary else episode.content
                            if episode.search_keywords:
                                content += " " + " ".join(episode.search_keywords)
                        else:  # Dict format
                            content = episode.get("summary", "") or episode.get("content", "")
                            keywords = episode.get("keywords", []) or episode.get("search_keywords", [])
                            if keywords:
                                content += " " + " ".join(keywords)
                        
                        documents.append(content if content else "")
                    
                    # Tokenize documents
                    tokenized_docs = []
                    for doc in documents:
                        tokens = self._tokenize_query(doc)
                        tokenized_docs.append(tokens)
                    
                    # Build BM25 index
                    if tokenized_docs and any(tokens for tokens in tokenized_docs):
                        bm25_index = BM25Okapi(tokenized_docs)
                        
                        # Update the index data and save it
                        index_data["bm25_index"] = bm25_index
                        index_data["corpus"] = tokenized_docs
                        index_data["documents"] = documents
                        index_data["bm25_available"] = True
                        
                        # Save updated index
                        with open(index_file, 'wb') as f:
                            pickle.dump(index_data, f)
                        
                        logger.info(f"Built BM25 index on-the-fly for user {owner_id}")
                    else:
                        return []
                        
                except Exception as e:
                    logger.error(f"Failed to build BM25 index on-the-fly for {owner_id}: {e}")
                    return []
            
            # Tokenize query using the same method used for indexing
            query_tokens = self._tokenize_query(query)
            
            if not query_tokens:
                return []
            
            # Get BM25 scores for all documents
            scores = bm25_index.get_scores(query_tokens)
            
            # Create (score, index) pairs and sort by score descending
            scored_docs = [(scores[i], i) for i in range(len(episodes))]
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            
            # Return top_k results with positive scores
            results = []
            for score, doc_idx in scored_docs[:top_k]:
                if score > 0:
                    # Extract content from the Nemori Episode object
                    episode = episodes[doc_idx]
                    content = episode.summary if episode.summary else episode.content
                    if content and content.strip():
                        results.append(content)
            
            return results
                
        except Exception as e:
            logger.error(f"BM25 search failed for user {owner_id}: {e}")
            return []
    
    def _tokenize_query(self, text: str) -> List[str]:
        """Tokenize query using the same method as index building."""
        if not text:
            return []
        
        text = text.lower()
        
        if self.nltk_available:
            try:
                from nltk.tokenize import word_tokenize
                tokens = word_tokenize(text)
                processed_tokens = []
                for token in tokens:
                    if token.isalpha() and len(token) >= 2 and token not in self.stop_words:
                        if self.stemmer:
                            stemmed_token = self.stemmer.stem(token)
                            processed_tokens.append(stemmed_token)
                        else:
                            processed_tokens.append(token)
                return processed_tokens
            except Exception:
                pass
        
        # Fallback to regex-based tokenization
        import re
        text = re.sub(r'[^\u4e00-\u9fff\w\s]', ' ', text)
        tokens = []
        current_word = ""
        for char in text:
            if '\u4e00' <= char <= '\u9fff':  # Chinese character
                if current_word.strip():
                    word = current_word.strip()
                    if len(word) >= 2:
                        tokens.append(word)
                    current_word = ""
                tokens.append(char)
            elif char.isalnum():  # English letter or digit
                current_word += char
            else:  # Space or other
                if current_word.strip():
                    word = current_word.strip()
                    if len(word) >= 2:
                        tokens.append(word)
                    current_word = ""
        if current_word.strip():
            word = current_word.strip()
            if len(word) >= 2:
                tokens.append(word)
        return tokens
    
    def get_available_users(self) -> List[str]:
        """Get list of users with available BM25 indices."""
        return list(self.available_users)
    
    def get_user_index_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a user's index.
        
        Args:
            user_id: User identifier
            
        Returns:
            Index metadata or None if not found
        """
        if user_id not in self.available_users:
            return None
        
        try:
            # Load user's index file to get metadata
            import pickle
            index_file = self.index_dir / f"bm25_index_{user_id}.pkl"
            
            if not index_file.exists():
                return None
            
            with open(index_file, 'rb') as f:
                index_data = pickle.load(f)
            
            return {
                "user_id": user_id,
                "document_count": len(index_data.get("episodes", [])),
                "corpus_size": len(index_data.get("corpus", [])),
                "created_at": index_data.get("created_at"),
                "tokenization_method": index_data.get("tokenization_method", "unknown"),
                "provider_type": "direct_bm25",
                "bm25_available": index_data.get("bm25_available", False)
            }
            
        except Exception as e:
            logger.error(f"Error getting index info for {user_id}: {e}")
            return None


class KeywordRetriever(BaseRetriever):
    """Keyword-based retriever implementation."""
    
    def __init__(self, version: str = "default", data_base: str = "memory_base/static_memory_base/bm25_index", user_id: str = None):
        """
        Initialize Keyword retriever.
        
        Args:
            data_base: Base path for memory storage
            version: Version identifier for the memory base
            user_id: User ID to filter documents by (optional)
        """
        self.version = version
        self.data_base = data_base
        self.user_id = user_id
        self.client = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize the retriever."""
        if self._initialized:
            return

        try:
            # 使用传入的版本名和 user_id
            self.client = KeywordSearch(version=self.version, output_dir=self.data_base, user_id=self.user_id)
            # 异步初始化，加载已存在的索引
            await self.client.initialize()
            self._initialized = True
        except Exception as e:
            logging.error(f"Failed to initialize Keyword retriever: {e}")
            raise
    
    async def search(self, query: str, top_k: int = 10, owner_id: str = "default") -> List[str]:
        """Search using Keyword."""
        if not self._initialized:
            await self.initialize()
        
        try:
            logging.info(f"KeywordRetriever searching for: '{query}' with top_k={top_k}")
            # Check if this is the dummy KeywordSearch class (async) or real one (sync)
            import inspect
            if inspect.iscoroutinefunction(self.client.search):
                # This is the dummy class, search is async and returns empty list
                logging.info("Using async KeywordSearch (dummy)")
                search_result = await self.client.search(query, top_k)
                if isinstance(search_result, list):
                    return search_result
                else:
                    results, scores = search_result
            else:
                # This is the real KeywordSearch class, search is sync and returns tuple
                logging.info("Using sync KeywordSearch (real)")
                results, scores = self.client.search(query, top_k)
                logging.info(f"KeywordRetriever found {len(results)} results")
            return results
        except Exception as e:
            logging.error(f"Keyword search failed: {e}")
            import traceback
            traceback.print_exc()
            return []


class VectorRetriever(BaseRetriever):
    """Vector-based retriever implementation."""
    
    def __init__(self, version: str = "default", data_base: str = "memory_base/static_memory_base/vector_search"):
        """
        Initialize Vector retriever.
        
        Args:
            data_base: Base path for memory storage
            version: Version identifier for the memory base
        """
        self.version = version
        self.data_base = data_base
        self.client = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize the retriever."""
        if self._initialized:
            return
        
        try:
            self.client = VectorSearch(version=self.version, output_dir=self.data_base)
            self._initialized = True
        except Exception as e:
            logging.error(f"Failed to initialize Vector retriever: {e}")
            raise

    async def search(self, query: str, top_k: int = 10, owner_id: str = "default") -> List[str]:
        """Search using Vector."""
        if not self._initialized:
            await self.initialize()
        
        try:
            results = self.client.search(query, top_k)
            content_results = [result['content'] for result in results]
            return content_results
        except Exception as e:
            logging.error(f"Vector search failed: {e}")
            return []


class MixedRetriever(BaseRetriever):
    """Mixed retriever that combines multiple methods with parallel execution."""
    
    def __init__(self, version: str = "default", data_base: str = "memory_base"):
        """
        Initialize mixed retriever.
        
        Args:
            version: Version identifier for the memory base
        """
        self.version = version
        self.data_base = data_base
        # Initialize retrievers
        self.keyword_retriever = None
        self.vector_retriever = None
        self.dynamic_memory_retriever = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize all retrievers with the specified version."""
        if self._initialized:
            return
            
        try:
            # Initialize keyword retriever
            self.keyword_retriever = KeywordRetriever(version=self.version, data_base=self.data_base, user_id=self.user_id)
            await self.keyword_retriever.initialize()
            
            # Initialize vector retriever
            self.vector_retriever = VectorRetriever(version=self.version, data_base=os.path.join(self.data_base, "static_memory_base", "vector_search"))
            await self.vector_retriever.initialize()
            
            # Initialize dynamic memory retriever
            self.dynamic_memory_retriever = DynamicMemoryRetriever(version=self.version, data_base=self.data_base)
            await self.dynamic_memory_retriever.initialize()
            
            self._initialized = True
        except Exception as e:
            logging.error(f"Failed to initialize mixed retriever: {e}")
            raise
    
    async def search(self, query: str, top_k: int = 10, owner_id: str = "default") -> List[str]:
        """Search using all retrieval methods in parallel and combine results."""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Execute all searches in parallel
            keyword_task = self.keyword_retriever.search(query, top_k, owner_id)
            vector_task = self.vector_retriever.search(query, top_k, owner_id)
            dynamic_memory_task = self.dynamic_memory_retriever.search(query, top_k, owner_id)
            
            # Wait for all to complete
            keyword_results, vector_results, dynamic_memory_results = await asyncio.gather(
                keyword_task, 
                vector_task,
                dynamic_memory_task,
                return_exceptions=True
            )
            
            # Handle exceptions from individual searches
            if isinstance(keyword_results, Exception):
                logging.error(f"Keyword search failed: {keyword_results}")
                keyword_results = []
            if isinstance(vector_results, Exception):
                logging.error(f"Vector search failed: {vector_results}")
                vector_results = []
            if isinstance(dynamic_memory_results, Exception):
                logging.error(f"Dynamic Memory search failed: {dynamic_memory_results}")
                dynamic_memory_results = []
            
            # Combine and weight results
            all_results = {
                "keyword": keyword_results,
                "vector": vector_results,
                "dynamic_memory": dynamic_memory_results
            }
            
            # Use RRF fusion to combine results
            final_results = self.rrf_fusion(all_results, top_k)
            
            return final_results
            
        except Exception as e:
            logging.error(f"Mixed search failed: {e}")
            return []
    
    @classmethod
    def rrf_fusion(cls, all_results: Dict[str, List[str]], top_k: int = 10) -> List[str]:
        """
        Perform Reciprocal Rank Fusion (RRF) to combine results from multiple retrievers.
        
        Args:
            all_results: Dictionary containing results from different retrievers
                Format: {"method_name": [doc1, doc2, ...], ...}
            top_k: Number of top results to return
            
        Returns:
            List of unique documents sorted by RRF score
        """
        # RRF constant (typically 60)
        k = 60
        
        # Dictionary to store document content -> RRF score
        doc_scores = {}
        
        # Process each retriever's results
        for method_name, docs in all_results.items():
            if not docs:  # Skip empty results
                continue
                
            for rank, doc in enumerate(docs):
                if doc not in doc_scores:
                    doc_scores[doc] = 0.0
                
                # Calculate RRF score: 1 / (k + rank)
                # rank is 0-indexed, so we add 1 to make it 1-indexed
                rrf_score = 1.0 / (k + rank + 1)
                doc_scores[doc] += rrf_score
        
        # Sort documents by RRF score in descending order
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top_k unique documents (only content, no scores)
        result_docs = []
        seen_docs = set()
        
        for doc, score in sorted_docs:
            if doc not in seen_docs:
                result_docs.append(doc)
                seen_docs.add(doc)
                if len(result_docs) >= top_k:
                    break
        
        return result_docs
    
    def get_method_info(self) -> Dict[str, Any]:
        """Get information about the mixed retrieval method."""
        return {
            "method": "mix",
            "version": self.version,
            "initialized": self._initialized
        }


class UnifiedRetriever:
    """Main retriever class that provides a unified interface."""
    
    def __init__(self, version: str = "default", method: str = "nemori", data_base: str = "memory_base", top_k: int = 10, user_id: str = None):
        """
        Initialize the unified retriever.
        
        Args:
            method: Retrieval method ("nemori", "hipporag", or "mix")
            data_base: Path to the memory storage
            top_k: Default number of results to return
            user_id: User ID to filter documents by (optional)
        """
        self.version = version
        self.method = method
        self.data_base = data_base
        self.top_k = top_k
        self.user_id = user_id
        self.retriever = None
        self._initialized = False
        
        # Validate method
        valid_methods = ["keyword", "vector", "dynamic_memory", "mix"]
        if method not in valid_methods:
            raise ValueError(f"Invalid method: {method}. Must be one of {valid_methods}")
    
    async def initialize(self):
        """Initialize the retriever based on the selected method."""
        if self._initialized:
            return

        try:
            if self.method == "keyword":
                self.retriever = KeywordRetriever(version=self.version, data_base=self.data_base, user_id=self.user_id)
            elif self.method == "vector":
                self.retriever = VectorRetriever(version=self.version, data_base=self.data_base)
            elif self.method == "dynamic_memory":
                self.retriever = DynamicMemoryRetriever(version=self.version, data_base=self.data_base)
            elif self.method == "mix":
                self.retriever = MixedRetriever(version=self.version, data_base=self.data_base)
            
            await self.retriever.initialize()
            print("retriever initialized")
            print("data_base",self.data_base)
            self._initialized = True
            
        except Exception as e:
            logging.error(f"Failed to initialize {self.method} retriever: {e}")
            raise
    
    async def retrieve(self, query: str, top_k: Optional[int] = None, owner_id: str = "default") -> List[str]:
        """
        Retrieve memories using the specified method.
        
        Args:
            query: Search query
            top_k: Number of results to return (overrides default)
            
        Returns:
            List of retrieved memory strings
        """
        if not self._initialized:
            await self.initialize()
        
        k = top_k if top_k is not None else self.top_k
        return await self.retriever.search(query, k, owner_id)
    
    def update_config(self, method: Optional[str] = None, top_k: Optional[int] = None):
        """
        Update retriever configuration.
        
        Args:
            method: New retrieval method
            data_base: New database path
            top_k: New default top_k value
        """
        if method and method != self.method:
            self.method = method
            self.retriever = None
            self._initialized = False
            
        if top_k is not None:
            self.top_k = top_k
    
    async def get_status(self) -> Dict[str, Any]:
        """Get retriever status information."""
        return {
            "method": self.method,
            "top_k": self.top_k,
            "initialized": self._initialized,
            "available_methods": ["keyword", "vector", "dynamic_memory", "mix"]
        }


# Convenience function for quick retrieval
async def retrieve_memories(
    query: str,
    method: str = "keyword",
    data_base: str = "memory_base",
    top_k: int = 10,
    owner_id: str = "default"
) -> List[str]:
    """
    Convenience function for quick memory retrieval.
    
    Args:
        query: Search query
        method: Retrieval method
        data_base: Database path
        top_k: Number of results
        owner_id: Owner ID for the memories
        
    Returns:
        List of retrieved memories
    """
    retriever = UnifiedRetriever(method=method, data_base=data_base, top_k=top_k)
    return await retriever.retrieve(query, top_k, owner_id)
