import os
import pickle
import logging
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from collections import OrderedDict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer
from .hipporag.embedding_model.LocalEmbedding import LocalEmbeddingModel
from .hipporag.embedding_model.OpenAI import OpenAIEmbeddingModel
from .hipporag.utils.config_utils import BaseConfig
from .hipporag.utils.misc_utils import compute_mdhash_id
from .faiss_index import FaissIVFPQ, generate_data_chunks

logger = logging.getLogger(__name__)

class VectorSearch:
    def __init__(self, version: str = "default", output_dir=None, max_tokens_len=512, num_workers=32, global_config=None):
        """
        Initialize the Embedding class for generating embeddings using embedding models.
        
        Args:
            model_name: Name of the embedding model to use ("local" or "text-embedding-ada-002")
            output_folder_path: Path to save the generated embeddings and chunks
            max_tokens_len: Maximum token length for text chunking
            num_workers: Number of workers for tokenization
            global_config: Global configuration object containing embedding settings
        """

        self.max_tokens_len = max_tokens_len
        self.num_workers = num_workers
        
        if not global_config:
            self.global_config = BaseConfig()
        else:
            self.global_config = global_config
        
        # Initialize paths
        self.embedding_model_name = self.global_config.embedding_model_name
        if output_dir is not None:
            self.storage_dir = Path(output_dir) / f"{version}_{self.embedding_model_name}"
        else:
            self.storage_dir = Path(f"memory_base/static_memory_base/vector_search/{version}_{self.embedding_model_name}")
        self.chunk_refs_path = self.storage_dir / "chunks" / "chunk_references_str.pkl"
        self.chunk_bias_path = self.storage_dir / "chunks" / "chunk_bias.pkl"
        self.embeddings_path = self.storage_dir / "embeddings" / "embeddings.parquet"
        self.index_path = self.storage_dir / "index.bin"
        self.metadata_path = self.storage_dir / "metadata.pkl"

        # Initialize faiss index if exists
        if os.path.exists(self.index_path):
            self.load_faiss_index()
        else:
            self.faiss_index = None
            self.metadata = None
            self._create_output_dirs()
        
        # Initialize embedding model
        self._init_embedding_model()
        
        # Initialize tokenizer
        self._init_tokenizer()
        
    def _init_embedding_model(self):
        """Initialize the embedding model based on the model name."""
        # Create a basic config for the embedding model
        if "text-embedding" in self.global_config.embedding_model_name:
            self.embedding_model = OpenAIEmbeddingModel(global_config=self.global_config)
        else:
            self.embedding_model = LocalEmbeddingModel(global_config=self.global_config)
            
        logger.info(f"Initialized embedding model: {self.global_config.embedding_model_name}")
        
    def _init_tokenizer(self):
        """Initialize the tokenizer for text chunking."""
        tokenizer_path = "/mnt/cxzx/share/model_checkpoints/gpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        logger.info("Initialized tokenizer")
        
    def _create_output_dirs(self):
        """Create output directories for saving embeddings and chunks."""
        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(self.storage_dir / "chunks", exist_ok=True)
        os.makedirs(self.storage_dir / "embeddings", exist_ok=True)
        logger.info(f"Created output directories: {self.storage_dir}")
            
    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        """Format instruction and query for embedding."""
        return f'Instruct: {task_description}\nQuery:{query}'
        
    def process_text(self, text):
        """Tokenize a single text chunk."""
        tokens = self.tokenizer.encode(text)
        return tokens
        
    def chunk_texts(self, unique_references):
        """Chunk texts based on token length and save chunked references."""
        print('Tokenizing texts...')
        
        # Tokenize all texts
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            unique_references_tokens = list(tqdm(
                executor.map(self.process_text, unique_references), 
                total=len(unique_references), 
                desc="Tokenizing texts"
            ))
        
        # Chunk texts based on token length and generate hash IDs
        chunk_unique_references_tokens = OrderedDict()
        bias = []
        idx = 0
        
        for tokens in tqdm(unique_references_tokens):
            if len(tokens) < self.max_tokens_len:
                token2str = self.tokenizer.decode(tokens)
                # Generate hash ID for this chunk
                chunk_hash = compute_mdhash_id(token2str)
                chunk_id = f"chunk-{chunk_hash}"
                chunk_unique_references_tokens[chunk_id] = token2str
                bias.append(len(chunk_unique_references_tokens))
            else:
                temp = []
                for i in range(0, len(tokens), self.max_tokens_len):
                    token2str = self.tokenizer.decode(tokens[i:i+self.max_tokens_len])
                    # Generate hash ID for this chunk
                    chunk_hash = compute_mdhash_id(token2str)
                    chunk_id = f"chunk-{chunk_hash}"
                    temp.append((chunk_id, token2str))
                
                # Add all chunks from this text
                for chunk_id, chunk_text in temp:
                    chunk_unique_references_tokens[chunk_id] = chunk_text
                bias.append(len(chunk_unique_references_tokens))
        
        # Save chunked references
        with open(self.chunk_refs_path, 'wb') as f:
            pickle.dump(chunk_unique_references_tokens, f)
        with open(self.chunk_bias_path, 'wb') as f:
            pickle.dump(bias, f)
        logger.info(f"Saved chunked references to {self.chunk_refs_path}")
            
        return chunk_unique_references_tokens
        
    def generate_embeddings(self, chunks):
        """Generate embeddings for a list of texts using batch_encode and save as parquet."""
        # Check if output file already exists and try to load from it
        if os.path.exists(self.embeddings_path):
            print(f"Found existing embeddings file: {self.embeddings_path}")
            print("Loading embeddings from existing file...")
            try:
                embeddings, contents, hash_ids = self.load_embeddings()
                print(f"Successfully loaded embeddings from {self.embeddings_path}")
                print(f"Loaded embeddings shape: {embeddings.shape}")
                return embeddings, contents, hash_ids
            except Exception as e:
                print(f"Failed to load existing embeddings: {str(e)}")
                print("Will generate new embeddings...")
        
        print(f"Generating embeddings for {len(chunks)} chunks...")
        
        try:
            # Handle both OrderedDict and list inputs
            if isinstance(chunks, OrderedDict):
                # If chunks is OrderedDict, extract contents for batch_encode
                chunk_contents = list(chunks.values())
                chunk_ids = list(chunks.keys())
            else:
                # If chunks is list, use as is and generate hash IDs
                chunk_contents = chunks
                chunk_ids = [f"chunk-{compute_mdhash_id(content)}" for content in chunks]
            
            # Use batch_encode to generate embeddings
            embeddings = self.embedding_model.batch_encode(chunk_contents)
            print(f"Generated embeddings shape: {embeddings.shape}")
            
            self.save_embeddings(embeddings, chunk_contents, chunk_ids)
            return embeddings, chunk_contents, chunk_ids
            
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            raise

    def save_embeddings(self, embeddings, chunk_contents, chunk_ids):
        """Save embeddings to parquet file."""
        df = pd.DataFrame({
            'hash_id': chunk_ids,
            'content': chunk_contents,
            'embedding': list(embeddings)  # Convert numpy arrays to list for parquet storage
        })
        df.to_parquet(self.embeddings_path, index=False)
        print(f"Saved embeddings to {self.embeddings_path}")

    def load_embeddings(self):
        """Load saved embeddings from parquet file."""
        try:
            df = pd.read_parquet(self.embeddings_path)
            print(f"Loaded embeddings from {self.embeddings_path}")
            print(f"Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            
            # Convert embeddings back to numpy arrays
            embeddings = np.array([np.array(emb) for emb in df['embedding']])
            contents = df['content'].tolist()
            hash_ids = df['hash_id'].tolist()
            
            return embeddings, contents, hash_ids
        
        except Exception as e:
            print(f"Error loading embeddings: {str(e)}")
            raise

    def train_faiss_index(self, documents):
        """Train Faiss index with embeddings and save it."""
        # Chunk documents
        chunks = self.chunk_texts(documents)

        # Generate embeddings
        embeddings, contents, hash_ids = self.generate_embeddings(chunks)
        
        dim = embeddings.shape[1]
        num_vectors = len(embeddings)
        
        print(f"Training Faiss index with {num_vectors} vectors of dimension {dim}")
        
        # nlist should not exceed the number of training vectors
        nlist = max(1, int(round(4 * np.sqrt(num_vectors))))
        
        # Reinitialize Faiss index with adjusted parameters
        faiss_config = {
            "dim": dim,
            "nlist": nlist,
            "nbits": 8,
            "nprobe": min(20, nlist)
        }
        self.faiss_index = FaissIVFPQ(config=faiss_config)
        
        # Create metadata with both content and hash ID
        metadata_with_ids = [(hash_ids[i], contents[i]) for i in range(len(hash_ids))]

        # Handle different index types
        if isinstance(self.faiss_index, faiss.IndexFlatIP):
            # Simple flat index - no training needed, just add vectors
            print("Adding vectors to flat index...")
            self.faiss_index.add(embeddings)
            
            # Store metadata separately since flat index doesn't support metadata
            self.metadata = {i: metadata_with_ids[i] for i in range(len(metadata_with_ids))}
        else:
            # Clustering-based index - needs training
            data_generator = generate_data_chunks(embeddings, metadata_with_ids)
            self.faiss_index.train(data_generator, total_samples=num_vectors)
            
            # Add vectors with metadata and custom IDs
            data_generator = generate_data_chunks(embeddings, metadata_with_ids)
            self.faiss_index.add_vectors_with_metadata(data_generator, total_samples=num_vectors, custom_ids=hash_ids)
        
        # Save the trained index
        if isinstance(self.faiss_index, faiss.IndexFlatIP):
            # Save simple flat index
            faiss.write_index(self.faiss_index, str(self.index_path))
            # Save metadata separately
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            print(f"Flat index saved to {self.index_path}")
        else:
            # Save clustering-based index
            self.faiss_index.save_index(self.index_path, self.metadata_path)
            print(f"Faiss index trained and saved to {self.index_path}")

    def load_faiss_index(self):
        """Load a trained Faiss index."""
        try:
            # Try to load as a simple flat index first
            import faiss
            self.faiss_index = faiss.read_index(str(self.index_path))
            
            # Check if it's a flat index and load metadata separately
            if isinstance(self.faiss_index, faiss.IndexFlatIP):
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                logger.info(f"Flat index loaded from {self.index_path}")
            else:
                # Load clustering-based index
                self.faiss_index = FaissIVFPQ()
                self.faiss_index.load_index(self.index_path, self.metadata_path)
                logger.info(f"Faiss index loaded from {self.index_path}")
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise

    def add_to_existing_index(self, documents):
        """Add new embeddings to an existing trained Faiss index."""
        # Chunk documents
        chunks = self.chunk_texts(documents)

        # Generate embeddings
        embeddings, contents, hash_ids = self.generate_embeddings(chunks)

        # Generate embeddings for the new chunks
        embeddings = self.embedding_model.batch_encode(contents)
        
        print(f"Adding {len(embeddings)} new vectors to existing index")
        
        # Create metadata with both content and hash ID
        metadata_with_ids = [(hash_ids[i], contents[i]) for i in range(len(hash_ids))]
        
        # Use Faiss.py's generate_data_chunks function directly
        data_generator = generate_data_chunks(embeddings, metadata_with_ids)
        
        # Add new vectors to the index with custom IDs
        self.faiss_index.add_vectors_with_metadata(data_generator, total_samples=len(embeddings), custom_ids=hash_ids)
        
        # Save updated index
        self.faiss_index.save_index(self.index_path, self.metadata_path)
        print(f"Updated index saved to {self.index_path}")

    def search(self, query, top_k=10):
        """
        Search for similar documents using a text query.
        
        Args:
            query (str): Text query to search for
            top_k (int): Number of top results to return
            
        Returns:
            list: List of dictionaries containing search results with format:
                - chunk_id: Document ID (chunk hash ID)
                - content: Original text content
                - score: Similarity score
        """
        if not hasattr(self, 'faiss_index') or self.faiss_index is None:
            raise ValueError("Faiss index not initialized. Please train or load an index first.")
        
        # Generate embedding for the query
        instruction = 'Given a search query, retrieve relevant passages that answer the query'
        query_embedding = self.embedding_model.batch_encode([query], instruction=instruction)
        
        # Handle different index types
        if isinstance(self.faiss_index, faiss.IndexFlatIP):
            # Simple flat index - search and get metadata separately
            similarities, indices = self.faiss_index.search(query_embedding, k=top_k)
            
            # Format results for flat index
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx in self.metadata:
                    chunk_id, content = self.metadata[idx]
                    formatted_result = {
                        'faiss_id': int(idx),
                        'chunk_id': chunk_id,
                        'content': content,
                        'score': float(similarity)
                    }
                    results.append(formatted_result)
            return results
        else:
            # Clustering-based index - use search_with_metadata
            search_results = self.faiss_index.search_with_metadata(query_embedding, k=top_k)
            
            # Format results
            results = []
            for result in search_results[0]:  # search_results[0] contains results for the single query
                # Extract chunk_id and content from metadata tuple
                if result['metadata']:
                    metadata = result['metadata']
                    
                    # Handle case where metadata might be None
                    if metadata is None:
                        print(f"Warning: metadata is None for result {result}")
                        continue
                    
                    formatted_result = {
                        'faiss_id': result['id'],
                        'chunk_id': metadata[0],
                        'content': metadata[1],
                        'score': result['cosine_similarity']
                    }
                    results.append(formatted_result)
            return results

    def get_all_id_to_rows(self):
        """Compatibility method to get all chunk data in the format expected by the original code."""
        if not hasattr(self, 'chunk_data') or self.chunk_data is None:
            # Load chunk data from saved files if available
            chunk_refs_path = os.path.join(self.chunk_folder_path, 'chunk_references_str.pkl')
            if os.path.exists(chunk_refs_path):
                with open(chunk_refs_path, 'rb') as f:
                    chunk_data = pickle.load(f)
                self.chunk_data = chunk_data
            else:
                self.chunk_data = {}
        
        # Convert to the expected format
        result = {}
        for chunk_id, content in self.chunk_data.items():
            result[chunk_id] = {'content': content}
        return result

    def get_all_texts(self):
        """Compatibility method to get all chunk texts."""
        if not hasattr(self, 'chunk_data') or self.chunk_data is None:
            self.get_all_id_to_rows()
        
        return list(self.chunk_data.values()) if self.chunk_data else []

    def get_row(self, chunk_id):
        """Compatibility method to get a specific chunk row."""
        if not hasattr(self, 'chunk_data') or self.chunk_data is None:
            self.get_all_id_to_rows()
        
        if chunk_id in self.chunk_data:
            return {'content': self.chunk_data[chunk_id]}
        return None

    def get_embeddings(self, chunk_ids):
        """Compatibility method to get embeddings for specific chunk IDs."""
        # Load embeddings from parquet file
        parquet_path = os.path.join(self.output_folder_path, "chunk_embeddings.parquet")
        if os.path.exists(parquet_path):
            df = pd.read_parquet(parquet_path)
            # Create a mapping from hash_id to embedding
            id_to_embedding = {}
            for idx, row in df.iterrows():
                id_to_embedding[row['hash_id']] = np.array(row['embedding'])
            
            # Return embeddings for requested chunk_ids
            embeddings = []
            for chunk_id in chunk_ids:
                if chunk_id in id_to_embedding:
                    embeddings.append(id_to_embedding[chunk_id])
                else:
                    # Return zero embedding if not found
                    embeddings.append(np.zeros(df['embedding'].iloc[0].shape))
            
            return embeddings
        else:
            # Return empty list if no embeddings file exists
            return []

    def get_all_ids(self):
        """Compatibility method to get all chunk IDs."""
        if not hasattr(self, 'chunk_data') or self.chunk_data is None:
            self.get_all_id_to_rows()
        
        return list(self.chunk_data.keys()) if self.chunk_data else []

    def text_to_hash_id(self, text):
        """Compatibility method to get hash ID for a given text."""
        if not hasattr(self, 'chunk_data') or self.chunk_data is None:
            self.get_all_id_to_rows()
        
        # Find the chunk ID for the given text
        for chunk_id, content in self.chunk_data.items():
            if content == text:
                return chunk_id
        return None

    def delete(self, chunk_ids):
        """Compatibility method to delete chunks by IDs."""
        if not hasattr(self, 'chunk_data') or self.chunk_data is None:
            self.get_all_id_to_rows()
        
        # Remove chunks from chunk_data
        for chunk_id in chunk_ids:
            if chunk_id in self.chunk_data:
                del self.chunk_data[chunk_id]
        
        # Update saved files
        if self.output_folder_path:
            chunk_refs_path = os.path.join(self.chunk_folder_path, 'chunk_references_str.pkl')
            with open(chunk_refs_path, 'wb') as f:
                pickle.dump(self.chunk_data, f)

