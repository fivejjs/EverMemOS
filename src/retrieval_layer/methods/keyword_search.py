"""
BM25 Index Utilities (Async Version)

This module provides asynchronous utilities for building and managing BM25 indices for document retrieval.
All major operations (index, search, save, load) are implemented as async methods to support
concurrent operations and better performance in async applications.

Key Features:
- Async document indexing and searching
- Concurrent search operations
- Async file I/O for index persistence
- DuckDB integration for document storage
- User-specific document filtering

Usage:
    # Create instance
    keyword_search = KeywordSearch(version="my_index", user_id="user123")
    
    # Initialize (load existing index if available)
    await keyword_search.initialize()
    
    # Index documents
    await keyword_search.index(documents)
    
    # Search documents
    docs, scores = await keyword_search.search("query", top_k=10)
    
    # Save index
    await keyword_search.save()
"""

import re
import logging
import pickle
import os
import asyncio
from pathlib import Path
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class KeywordSearch:
    """
    BM25 Index for document retrieval.
    
    This class provides functionality to build and manage a BM25 index for efficient
    document retrieval based on keyword matching.
    """
    
    def __init__(self, version: str = "default", output_dir=None, k1=1.2, b=0.75, user_id=None):
        """
        Initialize BM25 Index.
        
        Args:
            version: Version identifier for the memory base
            output_dir: Directory containing the index and documents
            k1: BM25 k1 parameter
            b: BM25 b parameter
            user_id: User ID to filter documents by (optional)
        """
        if output_dir is not None:
            self.storage_dir = Path(output_dir)
        else:
            self.storage_dir = Path(f"memory_base/dynamic_memory_base/storages")
        self.index_path = self.storage_dir / f"{version}.pkl"   
        self.documents_path = self.storage_dir / f"{version}.duckdb"
        self.user_id = user_id

        # 初始化基本属性
        self.bm25 = None
        self.documents = {}
        self.k1 = k1
        self.b = b
    
    async def initialize(self):
        """
        异步初始化方法，用于加载已存在的索引和文档。
        只有在索引文件存在时才自动加载。
        """
        if os.path.exists(self.index_path) and os.path.exists(self.documents_path):
            await self.load()
        
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text for indexing.
        
        Args:
            text (str): Input text to preprocess
            
        Returns:
            List[str]: List of preprocessed tokens
        """ 
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Split into tokens
        tokens = text.split()
        
        # Remove empty tokens
        tokens = [token for token in tokens if token.strip()]
        
        return tokens
    
    def min_max_normalize(self, scores: List[float]) -> List[float]:
        if not scores: return []
        if np.all(scores == 0): return scores
        
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        normalized_scores = (scores - min_score) / (max_score - min_score)
        return normalized_scores
    
    async def index(self, documents: List[str]):
        """
        Build BM25 index from a list of documents or a list of dictionaries.
        
        Args:
            documents: List of documents. Can be either:
                - List of strings (document contents)
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")
        
        # 在异步上下文中处理文档
        await asyncio.sleep(0)  # 让出控制权，允许其他协程运行
        
        self.documents = {i: doc for i, doc in enumerate(documents)}
        
        # 异步处理文档预处理
        tokenized_documents = []
        for doc in documents:
            tokenized_doc = self.preprocess_text(doc)
            tokenized_documents.append(tokenized_doc)
            # 每处理一定数量的文档后让出控制权
            if len(tokenized_documents) % 100 == 0:
                await asyncio.sleep(0)
        
        # Build BM25 index using rank_bm25
        self.bm25 = BM25Okapi(tokenized_documents, k1=self.k1, b=self.b)
        
        logger.info(f"BM25 index built successfully. Documents: {len(self.documents)}")
    
    async def search(self, query: str, top_k: int = 10) -> Tuple[List[str], List[float]]:
        """
        Search documents using BM25.
        
        Args:
            query (str): Search query
            top_k (int): Number of top results to return
            
        Returns:
            Tuple[List[str], List[float]]: Document IDs and scores of top results
        """
        if not self.bm25 or not self.documents:
            logger.warning("No BM25 index or documents available. Returning empty results.")
            return [], []
        
        # 让出控制权，允许其他协程运行
        await asyncio.sleep(0)
        
        # Preprocess query
        query_tokens = self.preprocess_text(query)
        
        if not query_tokens:
            logger.warning("Query preprocessing resulted in no tokens")
            return [], []
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Filter out results with zero scores
        non_zero_indices = np.where(scores > 0)[0]
        
        if len(non_zero_indices) == 0:
            logger.warning("No documents found with non-zero BM25 scores")
            return [], []
        
        # Sort by scores (descending) and get top-k results
        sorted_indices = non_zero_indices[np.argsort(scores[non_zero_indices])[::-1]]
        top_indices = sorted_indices[:top_k]
        
        # Return document IDs and scores
        docs = [self.documents[idx] for idx in top_indices]
        doc_scores = [scores[idx] for idx in top_indices]
        
        if doc_scores:
            normalized_scores = self.min_max_normalize(doc_scores)
        else:
            normalized_scores = []
        
        return docs, normalized_scores
    
    async def save(self):
        """
        Save the BM25 index and documents to files.
        Index is saved as pickle, documents are saved as DuckDB.
        """
        if not self.bm25:
            raise ValueError("Cannot save index that is not built")
        
        # 让出控制权，允许其他协程运行
        await asyncio.sleep(0)
        
        # Create directory if it doesn't exist
        if os.path.dirname(self.index_path):
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # Save BM25 index as pickle
        with open(self.index_path, 'wb') as f:
            pickle.dump(self.bm25, f)
        
        # Save documents as DuckDB
        await self._save_documents_to_duckdb()
        
        logger.info(f"BM25 index saved to {self.index_path}")
        logger.info(f"Documents saved to {self.documents_path}")
    
    async def _save_documents_to_duckdb(self):
        """
        Save documents to DuckDB format.
        Creates an 'episodes' table with the document data.
        """
        try:
            import duckdb
            
            # 让出控制权，允许其他协程运行
            await asyncio.sleep(0)
            
            # Connect to DuckDB file
            conn = duckdb.connect(str(self.documents_path))
             
            # Create episodes table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS episodes (
                    user_id VARCHAR,
                    user_name VARCHAR,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    title VARCHAR,
                    summary VARCHAR,
                    episode VARCHAR
                )
            """)
            
            # Clear existing data
            conn.execute("DELETE FROM episodes")
            
            # Insert documents
            # self.documents is a dictionary {index: document}
            for i, doc in self.documents.items():
                # Extract title and content from document
                # Assuming document format: "Title\n\nContent..."
                parts = doc.split('\n\n', 1)
                title = parts[0] if parts else f"Document {i+1}"
                content = parts[1] if len(parts) > 1 else doc
                
                # Generate summary (first 200 characters of content)
                summary = content[:200] + "..." if len(content) > 200 else content
                
                # Insert into database
                conn.execute("""
                    INSERT INTO episodes (user_id, user_name, timestamp, title, summary, episode)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, [
                    self.user_id or 'default',  # user_id
                    self.user_id or 'default',  # user_name (same as user_id for now)
                    None,  # timestamp (will use DEFAULT CURRENT_TIMESTAMP)
                    title,
                    summary,
                    content  # episode (full content)
                ])
                
                # 每处理一定数量的文档后让出控制权
                if (i + 1) % 50 == 0:
                    await asyncio.sleep(0)
            
            conn.close()
            logger.info(f"Successfully saved {len(self.documents)} documents to DuckDB")
            
        except Exception as e:
            logger.error(f"Failed to save documents to DuckDB: {e}")
            # Fallback to pickle if DuckDB fails
            with open(self.documents_path, 'wb') as f:
                pickle.dump(self.documents, f)
            logger.info(f"Fallback: documents saved as pickle to {self.documents_path}")
    
    async def load(self):
        """
        Load a BM25 index from a file.
        
        Args:
            filepath (str): Path to the saved index file
        """
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index file not found: {self.index_path}")
        
        if not os.path.exists(self.documents_path):
            raise FileNotFoundError(f"Documents file not found: {self.documents_path}")
        
        # 让出控制权，允许其他协程运行
        await asyncio.sleep(0)
        
        # Load BM25 index from pickle
        with open(self.index_path, 'rb') as f:
            bm25_data = pickle.load(f)
            
        # 如果加载的是字典格式的BM25数据，需要重新构建BM25对象
        if isinstance(bm25_data, dict):
            from rank_bm25 import BM25Okapi
            # 从字典中提取语料库
            corpus = bm25_data.get('corpus', [])
            # 重新构建BM25对象，使用默认参数
            self.bm25 = BM25Okapi(corpus, k1=1.2, b=0.75)
        else:
            self.bm25 = bm25_data

        # Load documents from file (could be DuckDB or pickle)
        try:
            # 首先尝试作为 DuckDB 文件加载
            import duckdb
            conn = duckdb.connect(str(self.documents_path))
            
            # 从 episodes 表中获取文档内容，根据 user_id 进行筛选
            if self.user_id:
                query = """
                SELECT user_id, user_name, timestamp, title, summary, episode
                FROM episodes 
                WHERE user_id = ?
                ORDER BY timestamp
                """
                rows = conn.execute(query, [self.user_id]).fetchall()
            else:
                query = """
                SELECT user_id, user_name, timestamp, title, summary, episode
                FROM episodes 
                ORDER BY timestamp
                """
                rows = conn.execute(query).fetchall()
            
            # 构建文档字典，每个文档包含标题和内容
            self.documents = {}
            for i, row in enumerate(rows):
                user_id, user_name, timestamp, title, summary, episode = row
                # 组合标题和episode内容作为文档
                doc_text = f"{title}\n{episode}"
                if summary and summary.strip():
                    doc_text += f"\n{summary}"
                self.documents[i] = doc_text
                
                # 每处理一定数量的文档后让出控制权
                if (i + 1) % 50 == 0:
                    await asyncio.sleep(0)
            
            conn.close()
            
            # 重新构建BM25索引，因为文档可能已经改变
            if self.documents:
                tokenized_documents = []
                for doc in self.documents.values():
                    tokenized_doc = self.preprocess_text(doc)
                    tokenized_documents.append(tokenized_doc)
                    # 每处理一定数量的文档后让出控制权
                    if len(tokenized_documents) % 100 == 0:
                        await asyncio.sleep(0)
                
                from rank_bm25 import BM25Okapi
                # 使用默认参数，因为此时 k1 和 b 可能还没有初始化
                self.bm25 = BM25Okapi(tokenized_documents, k1=1.2, b=0.75)
                logger.info(f"BM25 index rebuilt with {len(self.documents)} documents")
        except Exception as e:
            # 如果 DuckDB 加载失败，尝试作为 pickle 文件加载
            logger.info(f"DuckDB loading failed, trying pickle: {e}")
            try:
                with open(self.documents_path, 'rb') as f:
                    documents_data = pickle.load(f)
                
                if isinstance(documents_data, dict):
                    # 如果是字典格式，直接使用
                    self.documents = documents_data
                elif isinstance(documents_data, list):
                    # 如果已经是列表格式，转换为字典
                    self.documents = {i: doc for i, doc in enumerate(documents_data)}
                else:
                    logger.error(f"Unexpected documents data type: {type(documents_data)}")
                    self.documents = {}
            except Exception as e2:
                logger.error(f"Failed to load documents from {self.documents_path}: {e2}")
                self.documents = {}
        
        logger.info(f"BM25 index loaded from {self.index_path} and {len(self.documents)} documents loaded from {self.documents_path}")
