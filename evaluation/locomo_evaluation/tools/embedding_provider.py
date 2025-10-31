from typing import List
import requests
import numpy as np

class EmbeddingProvider:
    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url
        self.model_name = model_name

    def cosine_similarity(self, query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
        """
        Calculates cosine similarity between a query vector and multiple document vectors.

        Args:
            query_vec: A 1D numpy array for the query.
            doc_vecs: A 2D numpy array where each row is a document vector.

        Returns:
            A 1D numpy array of cosine similarity scores.
        """
        # Calculate dot product
        dot_product = np.dot(doc_vecs, query_vec)

        # Calculate norms
        query_norm = np.linalg.norm(query_vec)
        doc_norms = np.linalg.norm(doc_vecs, axis=1)

        # Calculate cosine similarity, handling potential division by zero
        denominator = query_norm * doc_norms
        # Replace 0s in denominator with a small number to avoid division by zero
        denominator[denominator == 0] = 1e-9
        
        similarity_scores = dot_product / denominator
        
        return similarity_scores
        
    def embed(self, texts: List[str]) -> str:
        if 'Qwen3' in self.model_name:
            response = requests.post(self.base_url, json={"input": texts, "model": self.model_name}).json()
        else:
            raise ValueError(f"Model {self.model_name} is not supported, only Qwen3-Reranker series models is supported")

        vectors = [item['embedding'] for item in response['data']]
        
        # scores = [item['score'] for item in response['data']]
        return vectors

if __name__ == "__main__":
    inputs = [
        "Tom上个月从他家乡搬过来了",
        "Frank的家乡是瑞士",
    ]

    reranker = EmbeddingProvider(base_url="http://0.0.0.0:11000/v1/embeddings", model_name="Qwen3-Embedding-4B")
    print(reranker.embed(inputs))