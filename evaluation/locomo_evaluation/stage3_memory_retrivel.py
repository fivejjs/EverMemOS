import json
import os
import sys
import pickle
from pathlib import Path
from typing import List, Tuple

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import asyncio
# Ensure project root is on sys.path so `evaluation` can be imported when running directly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
SRC_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_DIR)

from evaluation.locomo_evaluation.config import ExperimentConfig
# from evaluation.locomo_evaluation.tools.embedding_provider import EmbeddingProvider
# from evaluation.locomo_evaluation.tools.reranker_provider import RerankerProvider
from src.agentic_layer import vectorize_service
from src.agentic_layer import rerank_service

# This file depends on the rank_bm25 library.
# If you haven't installed it yet, run: pip install rank_bm25
TEMPLATE = """Episodes memories for conversation between {speaker_1} and {speaker_2}:

    {speaker_memories}
"""
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


def cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
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

def tokenize(text: str, stemmer, stop_words: set) -> list[str]:
    """
    NLTK-based tokenization with stemming and stopword removal.
    This must be identical to the tokenization used during indexing.
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

def search_with_bm25_index(query: str, bm25, docs, top_n: int = 5):
    """
    Performs BM25 search using a pre-loaded index.
    """
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    tokenized_query = tokenize(query, stemmer, stop_words)
    
    if not tokenized_query:
        print("Warning: Query is empty after tokenization.")
        return []

    doc_scores = bm25.get_scores(tokenized_query)
    results_with_scores = list(zip(docs, doc_scores))
    sorted_results = sorted(results_with_scores, key=lambda x: x[1], reverse=True)
    return sorted_results[:top_n]

async def search_with_emb_index(query: str, emb_index, top_n: int = 5):
    """
    Performs embedding search using a pre-loaded index.
    """
    query_vec = np.array(await vectorize_service.get_text_embedding(query))
    episode_embeddings = []
    docs_with_episode = []
    
    for item in emb_index:

        # if "episode" in item.get("embeddings", {}):
        episode_embeddings.append(item["embeddings"]["event_log_atomic_fact"])
        docs_with_episode.append(item["doc"])
    
    if not episode_embeddings:
        return []

    episode_embeddings_np = np.array(episode_embeddings)
    scores = cosine_similarity(query_vec, episode_embeddings_np)
    results_with_scores = list(zip(docs_with_episode, scores))
    sorted_results = sorted(results_with_scores, key=lambda x: x[1], reverse=True)
    return sorted_results[:top_n]


async def reranker_search(query: str, results: List[Tuple[dict, float]], top_n: int = 5, reranker_instruction: str = None):
    """
    Rerank search results using a reranker model.
    
    For documents with event_log, uses format: "time：atomic_fact1 atomic_fact2 ..."
    For documents without event_log, falls back to episode text.
    """
    if not results:
        return []

    docs_with_episode = []
    doc_texts = []
    for doc, score in results:
        # 优先使用event_log格式化文本（如果存在）
        if doc.get("event_log") and doc["event_log"].get("atomic_fact"):
            event_log = doc["event_log"]
            time_str = event_log.get("time", "")
            atomic_facts = event_log.get("atomic_fact", [])
            
            if isinstance(atomic_facts, list) and atomic_facts:
                # 格式: "time：atomic_fact1 atomic_fact2 ..."
                facts_text = " ".join(atomic_facts)
                if time_str:
                    formatted_text = f"{time_str}：{facts_text}"
                else:
                    formatted_text = facts_text
                
                docs_with_episode.append(doc)
                doc_texts.append(formatted_text)
                continue
        
        # 回退到原有的episode字段（保持向后兼容）
        if episode_text := doc.get("episode"):
            docs_with_episode.append(doc)
            doc_texts.append(episode_text)

    if not doc_texts:
        return []
    
    # The reranker API expects a list of queries, one for each doc
    queries = [query] * len(doc_texts)
    
    # rerank_scores = await rerank_service.rerank(queries, doc_texts)
    reranker = rerank_service.get_rerank_service()
    # rerank_results = await reranker._make_rerank_request(queries, doc_texts, instruction=reranker_instruction)
    print(f"Reranking query: {query}")
    # print(f"Reranking doc_texts: {doc_texts}")
    print(f"Reranking reranker_instruction: {reranker_instruction}")
    rerank_results = await reranker._make_rerank_request(query, doc_texts, instruction=reranker_instruction)
    
    sorted_results = [(results[item["index"]][0], item["relevance_score"]) for item in rerank_results["results"]]
    return sorted_results[:top_n]


async def main():
    """Main function to perform batch search and save results in nemori format."""
    # --- Configuration ---
    config = ExperimentConfig()
    bm25_index_dir = Path(__file__).parent / "results" / config.experiment_name / "bm25_index"
    emb_index_dir = Path(__file__).parent / "results" / config.experiment_name / "vectors"
    save_dir = Path(__file__).parent / "results" / config.experiment_name 
    
    dataset_path = config.datase_path
    results_output_path = save_dir / "search_results.json"
    
    # Ensure NLTK data is ready
    ensure_nltk_data()

    # Load the dataset
    print(f"Loading dataset from: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    all_search_results = {}

    
    # Iterate through the dataset, assuming the index of the dataset list
    # corresponds to the conversation index number.
    for i, conversation_data in enumerate(dataset):
        conv_id = f"locomo_exp_user_{i}"

        speaker_a = conversation_data["conversation"].get("speaker_a")
        speaker_b = conversation_data["conversation"].get("speaker_b")
        print(f"\n--- Processing Conversation ID: {conv_id} ---")
        
        if "qa" not in conversation_data:
            print(f"Warning: No 'qa' key found in conversation #{i}. Skipping.")
            continue
        
        # --- Load index once per conversation ---
        if config.use_emb:
            emb_index_path = emb_index_dir / f"embedding_index_conv_{i}.pkl"
            if not emb_index_path.exists():
                print(f"Error: Index file not found at {emb_index_path}. Skipping conversation.")
                continue
            with open(emb_index_path, "rb") as f:
                emb_index = pickle.load(f)
        else:
            bm25_index_path = bm25_index_dir / f"bm25_index_conv_{i}.pkl"
            if not bm25_index_path.exists():
                print(f"Error: Index file not found at {bm25_index_path}. Skipping conversation.")
                continue
            with open(bm25_index_path, "rb") as f:
                index_data = pickle.load(f)
            bm25 = index_data["bm25"]
            docs = index_data["docs"]
            
        # Parallelize per-question retrieval with bounded concurrency
        sem = asyncio.Semaphore(128)

        async def process_single_qa(qa_pair):
            question = qa_pair.get("question")
            if not question:
                return None
            if qa_pair.get("category") == 5:
                print(f"Skipping question {question} because it is category 5")
                return None
            try:
                async with sem:
                    # Perform the search for the current question in the corresponding index
                    if config.use_reranker:
                        if config.use_emb:
                            results = await search_with_emb_index(
                                query=question,
                                emb_index=emb_index,
                                top_n=100,
                            )
                        else:
                            results = await asyncio.to_thread(
                                search_with_bm25_index,
                                question,
                                bm25,
                                docs,
                                100,
                            )
                        top_results = await reranker_search(
                            query=question,
                            results=results,
                            top_n=5,
                            reranker_instruction=config.reranker_instruction,
                        )
                    else:
                        if config.use_emb:
                            top_results = await search_with_emb_index(
                                query=question,
                                emb_index=emb_index,
                                top_n=20,
                            )
                        else:
                            top_results = await asyncio.to_thread(
                                search_with_bm25_index,
                                question,
                                bm25,
                                docs,
                                20,
                            )
                    
                    # The "context" in nemori's output is a formatted string of retrieved docs.
                    # We will replicate this by combining the retrieved documents into a single string.
                    # 优先使用event_log格式，回退到原有格式
                    context_str = ""
                    if top_results:
                        retrieved_docs_text = []
                        for doc, score in top_results:
                            # 优先使用event_log格式化文本（如果存在）
                            if doc.get("event_log") and doc["event_log"].get("atomic_fact"):
                                event_log = doc["event_log"]
                                time_str = event_log.get("time", "N/A")
                                atomic_facts = event_log.get("atomic_fact", [])
                                
                                if isinstance(atomic_facts, list) and atomic_facts:
                                    # 格式化为: "Time: ...\nFacts:\n- fact1\n- fact2\n..."
                                    facts_list = "\n- ".join(atomic_facts)
                                    doc_text = f"{doc.get('subject', 'N/A')}\nTime: {time_str}\nFacts:\n- {facts_list}\n---"
                                    retrieved_docs_text.append(doc_text)
                                    continue
                            
                            # 回退到原有格式（保持向后兼容）
                            doc_text = f"{doc.get('subject', 'N/A')}: {doc.get('episode', 'N/A')}\n---"
                            retrieved_docs_text.append(doc_text)
                        context_str = "\n\n".join(retrieved_docs_text)

                    return {
                        "query": question,
                        "context": TEMPLATE.format(
                                speaker_1=speaker_a,
                                speaker_2=speaker_b,
                                speaker_memories=context_str
                            ),
                        # Adding original QA pair for easier evaluation if needed
                        "original_qa": qa_pair 
                    }
            except Exception as e:
                print(f"Error processing question '{question}': {e}")
                return None

        tasks = [asyncio.create_task(process_single_qa(qa_pair)) for qa_pair in conversation_data["qa"]]
        results_for_conv = [res for res in await asyncio.gather(*tasks) if res is not None]
        
        all_search_results[conv_id] = results_for_conv

    # Save all results to a single JSON file in the specified format
    print(f"\nSaving all retrieval results to: {results_output_path}")
    with open(results_output_path, "w", encoding="utf-8") as f:
        json.dump(all_search_results, f, indent=2, ensure_ascii=False)
        
    print("Batch search and retrieval complete!")

    # Clean up resources
    reranker = rerank_service.get_rerank_service()
    # Assuming the service is DeepInfraRerankService, which has a close method.
    if hasattr(reranker, 'close') and callable(getattr(reranker, 'close')):
        await reranker.close()


if __name__ == "__main__":
    asyncio.run(main())
