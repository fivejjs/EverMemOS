import os
import numpy as np
import faiss
import pickle
import time
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
from .hipporag.utils.misc_utils import min_max_normalize

faiss_config = {
    "dim": 768,
    "nlist": 1000,
    "m": 512,
    "nbits": 8,
    "nprobe": 20,
}

class FaissIVFPQ:
    """Faiss IndexIVFPQ retrieval system supporting cosine similarity"""
    
    def __init__(self, config: dict = faiss_config, use_gpu: bool = False):
        """Initialize the retrieval system"""
        self.dim = config.get("dim", 768)
        self.nlist = config.get("nlist", 1000)
        self.m = config.get("m", 8)
        self.nbits = config.get("nbits", 8)
        self.nprobe = config.get("nprobe", 10)
        self.use_gpu = use_gpu

        # For cosine similarity, we need to use inner product (IP) as distance metric
        # Note: All vectors must be L2 normalized before use
        self.quantizer = faiss.IndexFlatIP(self.dim)  # Use inner product quantizer
        
        # Delay index creation until training to determine optimal type based on data size
        self.index = None
        self.index_type = None  # 'ivfpq' or 'flatip'
        
        # if self.use_gpu:
        #     index_flat = faiss.IndexIVFPQ(self.quantizer, dim, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
        #     gpu_index_flat = faiss.index_cpu_to_all_gpus(index_flat)
        #     self.index = gpu_index_flat   

        # Metadata storage
        self.metadata: Dict[int, dict] = {}
        self.current_id = 0

        # GPU setup will be handled after index creation
        self._gpu_setup_pending = self.use_gpu
    
    def _setup_gpu(self):
        """Setup GPU acceleration"""
        res = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
    
    @staticmethod
    def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
        """
        Perform L2 normalization on vectors
        
        After normalization, inner product is equivalent to cosine similarity
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.maximum(norms, 1e-10)
        return vectors / norms
    
    def train(self, data_generator, total_samples: int, batch_size: int = 100000):
        if self.index is not None and self.index.is_trained:
            print("Index already trained, skipping training step")
            return

        print(f"Starting index training, total samples: {total_samples}")
        training_data = []

        for batch in tqdm(data_generator, desc="Loading training data"):
            vectors = batch if isinstance(batch, np.ndarray) else batch[0]
            training_data.append(vectors)
            if sum(len(b) for b in training_data) >= total_samples:
                break

        training_data = np.vstack(training_data).astype(np.float32, copy=False)
        print('Train Data Shape: ', training_data.shape)

        if len(training_data) > total_samples:
            idx = np.random.choice(len(training_data), total_samples, replace=False)
            training_data = training_data[idx]

        # TODO: use IndexFlatIP for all cases for now, need to fine tune parameters
        if True:
            print(f"Data size {total_samples} < 1000, using IndexFlatIP (no training needed)")
            base = faiss.IndexFlatIP(self.dim)
            # To support add_with_ids, we need to wrap the index with IDMap2
            self.index = faiss.IndexIDMap2(base)
            self.index_type = 'flatip'
        else:
            print(f"Data size {total_samples} >= 1000, using IndexIVFPQ with training")
            base = faiss.IndexIVFPQ(self.quantizer, self.dim, self.nlist, self.m, self.nbits, faiss.METRIC_INNER_PRODUCT)
            base.nprobe = self.nprobe

            print('Starting index training')
            t0 = time.time()
            base.train(training_data)
            print(f"Training completed, time taken: {time.time() - t0:.2f} seconds")
            
            # Wrap the index with IDMap2 to support add_with_ids
            self.index = faiss.IndexIDMap2(base)
            self.index_type = 'ivfpq'

        if self._gpu_setup_pending:
            self._setup_gpu()
            self._gpu_setup_pending = False

    def add_vectors_with_metadata(self, data_generator, total_samples: int, batch_size: int = 100000, custom_ids=None):
        if self.index is None:
            raise ValueError("Index not initialized, please call train method first")
        if self.index_type == 'ivfpq' and not self.index.is_trained:
            raise ValueError("Index not trained yet, please call train method first")

        print(f"Starting to add vectors and metadata, total samples: {total_samples}")
        added_count = 0

        for batch in tqdm(data_generator, desc="Adding vectors and metadata"):
            vectors, metadatas = batch
            vectors = np.asarray(vectors, dtype=np.float32, order="C")
            if len(vectors) != len(metadatas):
                raise ValueError(f"Vector count({len(vectors)}) doesn't match metadata count({len(metadatas)})")

            ids = np.arange(self.current_id, self.current_id + len(vectors), dtype=np.int64)

            for i, faiss_id in enumerate(ids):
                self.metadata[faiss_id] = metadatas[i]
            if custom_ids is not None:
                if not hasattr(self, 'chunk_id_mapping'):
                    self.chunk_id_mapping = {}
                for faiss_id, chunk_id in zip(ids, custom_ids[added_count:added_count + len(vectors)]):
                    self.chunk_id_mapping[faiss_id] = chunk_id

            self.index.add_with_ids(vectors, ids)

            self.current_id += len(vectors)
            added_count += len(vectors)
            if added_count >= total_samples:
                break

        print(f"Vector and metadata addition completed, index now contains {self.index.ntotal} vectors")

    def search_with_metadata(self, query_vectors: np.ndarray, k: int = 10) -> List[List[Dict]]:
        """
        Search for nearest neighbors and return cosine similarity
        
        Note: Query vectors will be automatically normalized
        """
        if self.index is None or self.index.ntotal == 0:
            raise ValueError("No vectors in index, please add vectors first")
        
        # Normalize query vectors
        normalized_queries = self.normalize_vectors(query_vectors)
        
        # Perform search (returns inner product, equivalent to cosine similarity when normalized)
        similarities, indices = self.index.search(normalized_queries, k)
        similarities = min_max_normalize(similarities)
        
        # Build results with cosine similarity
        results = []
        for i in range(len(normalized_queries)):
            query_results = []
            for sim, idx in zip(similarities[i], indices[i]):
                # Cosine similarity ranges from [-1, 1], higher values indicate more similarity
                metadata = self.metadata.get(idx, None)
                
                result = {
                    "cosine_similarity": float(sim),
                    "id": int(idx),
                    "metadata": metadata
                }
                
                # If chunk hash ID mapping exists, add to results
                if hasattr(self, 'chunk_id_mapping') and idx in self.chunk_id_mapping:
                    result["chunk_hash_id"] = self.chunk_id_mapping[idx]
                
                if result["metadata"] is None:
                    result["warning"] = "Metadata does not exist"
                
                query_results.append(result)
            results.append(query_results)
        
        return results

    def save_index(self, index_path: str, metadata_path: Optional[str] = None):
        """Save index and metadata"""
        if metadata_path is None:
            metadata_path = os.path.splitext(index_path)[0] + "_metadata.pkl"
        
        # Save index
        if self.use_gpu and self.index is not None:
            index = faiss.index_gpu_to_cpu(self.index)
        elif self.index is not None:
            index = self.index
        else:
            raise ValueError("Index not initialized or GPU setup failed.")
        
        faiss.write_index(index, str(index_path))
        print(f"Index saved to {index_path}")
        
        # Save metadata, including Faiss parameters and chunk ID mapping
        metadata_to_save = {
            'metadata': self.metadata,
            'current_id': self.current_id,
            'index_type': self.index_type,
            'faiss_params': {
                'dim': self.dim,
                'nlist': self.nlist,
                'm': self.m,
                'nbits': self.nbits,
                'nprobe': self.nprobe,
                'use_gpu': self.use_gpu
            }
        }
        
        # Save chunk ID mapping (if exists)
        if hasattr(self, 'chunk_id_mapping'):
            metadata_to_save['chunk_id_mapping'] = self.chunk_id_mapping
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata_to_save, f)
        print(f"Metadata saved to {metadata_path}")
    
    def load_index(self, index_path: str, metadata_path: Optional[str] = None):
        """Load index and metadata"""
        if metadata_path is None:
            metadata_path = os.path.splitext(index_path)[0] + "_metadata.pkl"
        
        # Load index
        self.index = faiss.read_index(str(index_path))
        if self.use_gpu and self.index is not None:
            self._setup_gpu()
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.metadata = data['metadata']
            self.current_id = data['current_id']
            
            # Restore index type
            self.index_type = data.get('index_type', 'ivfpq')  # Default to ivfpq for backward compatibility
            
            # Restore Faiss parameters
            if 'faiss_params' in data:
                faiss_params = data['faiss_params']
                self.dim = faiss_params['dim']
                self.nlist = faiss_params['nlist']
                self.m = faiss_params['m']
                self.nbits = faiss_params['nbits']
                self.nprobe = faiss_params['nprobe']
                self.use_gpu = faiss_params['use_gpu']
                # Set nprobe only for IVFPQ index
                if self.index_type == 'ivfpq' and self.index is not None:
                    self.index.nprobe = self.nprobe
            
            # Restore chunk ID mapping (if exists)
            if 'chunk_id_mapping' in data:
                self.chunk_id_mapping = data['chunk_id_mapping']
        
        print(f"Index loaded from {index_path}, contains {self.index.ntotal} vectors")
        print(f"Metadata loaded from {metadata_path}, contains {len(self.metadata)} records")
        if 'faiss_params' in data:
            print(f"Faiss parameters restored: dim={self.dim}, nlist={self.nlist}, m={self.m}, nbits={self.nbits}, nprobe={self.nprobe}, use_gpu={self.use_gpu}")
        if 'chunk_id_mapping' in data:
            print(f"Chunk ID mapping restored, contains {len(self.chunk_id_mapping)} mappings")



def generate_data_chunks(all_embeds, all_texts, batch_size: int = 100000):
    # Generate data in batches
    for i in range(0, len(all_embeds), batch_size):
        yield all_embeds[i:i+batch_size], all_texts[i:i+batch_size]


def evaluate_retrieval_system(system, query_vectors: np.ndarray, 
                              ground_truth: np.ndarray, k: int = 10) -> float:
    """
    Evaluate retrieval system performance
    
    Parameters:
        system: Retrieval system instance
        query_vectors: Query vectors
        ground_truth: True nearest neighbor indices
        k: Number of neighbors to evaluate
    
    Returns:
        Accuracy (Recall@k)
    """
    # Perform search
    distances, indices = system.search(query_vectors, k)
    
    # Calculate Recall@k
    correct = 0
    for i in range(len(query_vectors)):
        if ground_truth[i] in indices[i]:
            correct += 1
    
    recall = correct / len(query_vectors)
    print(f"Recall@{k}: {recall:.4f}")
    
    return recall

def load_data(embeds_path):
    # Load entire dataset
    all_embeds = []
    all_texts = []
    files = [fil for fil in os.listdir(embeds_path) if fil.endswith('.npy')]
    for fil in tqdm(files, total=len(files)):
        data = np.load(os.path.join(embeds_path, fil), allow_pickle=True).item()
        embeds = data['embeds']
        texts = data['texts']
        all_texts.extend(texts)
        all_embeds.append(embeds)
    all_embeds = np.concatenate(all_embeds)

    assert len(all_embeds) == len(all_texts)
    # print(f"L2 norm of first vector loaded from .npy files: {np.linalg.norm(all_embeds[0])}")
    return all_embeds, all_texts, len(all_embeds)
