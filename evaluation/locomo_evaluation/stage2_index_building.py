import json
import os
import sys
import pickle
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
import asyncio

# Ensure project root is on sys.path so `evaluation` can be imported when running directly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
SRC_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_DIR)


from evaluation.locomo_evaluation.config import ExperimentConfig
from evaluation.locomo_evaluation.tools.embedding_provider import EmbeddingProvider
from src.agentic_layer import vectorize_service




def ensure_nltk_data():
    """Ensure required NLTK data is downloaded."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

def build_searchable_text(doc: dict) -> str:
    """
    Build searchable text from a document with weighted fields.
    
    Priority:
    1. If event_log exists, use atomic_fact for indexing
    2. Otherwise, fall back to original fields:
       - "subject" corresponds to "title" (weight * 3)
       - "summary" corresponds to "summary" (weight * 2)
       - "episode" corresponds to "content" (weight * 1)
    """
    parts = []
    
    # 优先使用event_log的atomic_fact（如果存在）
    if doc.get("event_log") and doc["event_log"].get("atomic_fact"):
        atomic_facts = doc["event_log"]["atomic_fact"]
        if isinstance(atomic_facts, list):
            # 将所有atomic_fact拼接，每个fact重复一次（保持一致性）
            parts.extend(atomic_facts)
            return " ".join(str(fact) for fact in parts if fact)
    
    # 回退到原有字段（保持向后兼容）
    # Title has highest weight (repeat 3 times)
    if doc.get("subject"):
        parts.extend([doc["subject"]] * 3)

    # Summary (repeat 2 times)
    if doc.get("summary"):
        parts.extend([doc["summary"]] * 2)

    # Content
    if doc.get("episode"):
        parts.append(doc["episode"])

    return " ".join(str(part) for part in parts if part)


def tokenize(text: str, stemmer, stop_words: set) -> list[str]:
    """
    NLTK-based tokenization with stemming and stopword removal.
    """
    if not text:
        return []

    tokens = word_tokenize(text.lower())
    
    processed_tokens = [
        stemmer.stem(token) 
        for token in tokens 
        if token.isalpha() and len(token) >= 2 and token not in stop_words
    ]
    
    return processed_tokens

def build_bm25_index(config: ExperimentConfig, data_dir: Path, bm25_save_dir: Path) -> list[list[float]]:
# --- NLTK Setup ---
    print("Ensuring NLTK data is available...")
    ensure_nltk_data()
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    
    # --- Data Loading and Processing ---
    # corpus = [] # This line is removed as per the new_code
    # original_docs = [] # This line is removed as per the new_code
    
    print(f"Reading data from: {data_dir}")
    
    for i in range(config.num_conv):
        file_path = data_dir / f"memcell_list_conv_{i}.json"
        if not file_path.exists():
            print(f"Warning: File not found, skipping: {file_path}")
            continue
            
        print(f"\nProcessing {file_path.name}...")
        
        corpus = []
        original_docs = []

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
            for doc in data:
                original_docs.append(doc)
                searchable_text = build_searchable_text(doc)
                tokenized_text = tokenize(searchable_text, stemmer, stop_words)
                corpus.append(tokenized_text)

        if not corpus:
            print(f"Warning: No documents found in {file_path.name}. Skipping index creation.")
            continue
            
        print(f"Processed {len(original_docs)} documents from {file_path.name}.")

        # --- BM25 Indexing ---
        print(f"Building BM25 index for {file_path.name}...")
        bm25 = BM25Okapi(corpus)

        # --- Saving the Index ---
        index_data = {
            "bm25": bm25,
            "docs": original_docs
        }

        output_path = bm25_save_dir / f"bm25_index_conv_{i}.pkl"
        print(f"Saving index to: {output_path}")
        with open(output_path, "wb") as f:
            pickle.dump(index_data, f)
            
async def build_emb_index(config: ExperimentConfig, data_dir: Path, emb_save_dir: Path):
    # embedding_service = vector_service.get_vector_service()
    BATCH_SIZE = 256

    for i in range(config.num_conv):
        file_path = data_dir / f"memcell_list_conv_{i}.json"
        if not file_path.exists():
            print(f"Warning: File not found, skipping: {file_path}")
            continue
            
        print(f"\nProcessing {file_path.name} for embedding...")
        
        with open(file_path, "r", encoding="utf-8") as f:
            original_docs = json.load(f)

        texts_to_embed = []
        doc_field_map = []
        for doc_idx, doc in enumerate(original_docs):
            # 优先使用event_log（如果存在）
            if doc.get("event_log") and doc["event_log"].get("atomic_fact"):
                atomic_facts = doc["event_log"]["atomic_fact"]
                if isinstance(atomic_facts, list) and atomic_facts:
                    # 将所有atomic_fact拼接成一个文本用于embedding
                    combined_text = " ".join(atomic_facts)
                    texts_to_embed.append(combined_text)
                    doc_field_map.append((doc_idx, "event_log_atomic_fact"))
                    continue
            
            # 回退到原有字段（保持向后兼容）
            for field in ["subject", "summary", "episode"]:
                if text := doc.get(field):
                    texts_to_embed.append(text)
                    doc_field_map.append((doc_idx, field))

        if not texts_to_embed:
            print(f"Warning: No documents found in {file_path.name}. Skipping embedding creation.")
            continue
            
        print(f"Generating embeddings for {len(texts_to_embed)} text pieces from {file_path.name}...")
        
        all_embeddings = []
        for j in range(0, len(texts_to_embed), BATCH_SIZE):
            batch_texts = texts_to_embed[j:j+BATCH_SIZE]
            print(f"  - Processing batch {j//BATCH_SIZE + 1} ({len(batch_texts)} items)")
            # batch_embeddings = embedding_provider.embed(batch_texts)
            batch_embeddings = await vectorize_service.get_text_embeddings(batch_texts)
            all_embeddings.extend(batch_embeddings)

        # Re-associate embeddings with their original documents and fields
        doc_embeddings = [{} for _ in original_docs]
        for (doc_idx, field), emb in zip(doc_field_map, all_embeddings):
            if "embeddings" not in doc_embeddings[doc_idx]:
                doc_embeddings[doc_idx]["embeddings"] = {}
            doc_embeddings[doc_idx]["embeddings"][field] = emb
            doc_embeddings[doc_idx]["doc"] = original_docs[doc_idx]


        # The final structure of the saved .pkl file will be a list of dicts:
        # [
        #     {
        #         "doc": { ... original document ... },
        #         "embeddings": {
        #             "subject": [ ... embedding vector ... ],
        #             "summary": [ ... embedding vector ... ],
        #             "episode": [ ... embedding vector ... ]
        #         }
        #     },
        #     ...
        # ]
        output_path = emb_save_dir / f"embedding_index_conv_{i}.pkl"
        emb_save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving embeddings to: {output_path}")
        with open(output_path, "wb") as f:
            pickle.dump(doc_embeddings, f)
            

async def main():
    """Main function to build and save the BM25 index."""
    # --- Configuration ---
    # The directory containing the JSON files
    config = ExperimentConfig()
    data_dir = Path(__file__).parent / "results" / config.experiment_name / "memcells"
    bm25_save_dir = Path(__file__).parent / "results" / config.experiment_name / "bm25_index"
    emb_save_dir = Path(__file__).parent / "results" / config.experiment_name / "vectors"
    os.makedirs(bm25_save_dir, exist_ok=True)
    os.makedirs(emb_save_dir, exist_ok=True)
    build_bm25_index(config, data_dir, bm25_save_dir)
    if config.use_emb:
        await build_emb_index(config, data_dir, emb_save_dir)
    # data_dir = Path("/Users/admin/Documents/Projects/b001-memsys/evaluation/locomo_evaluation/results/locomo_evaluation_0/")
    
    # Where to save the final index file
    # output_path = data_dir / "bm25_index.pkl" # This line is removed as per the new_code
    
    
        
    print("\nAll indexing complete!")

if __name__ == "__main__":
    asyncio.run(main())
