"""
Rerank module for memory retrieval system.
基于Qwen3-Reranker模型的memory重排序模块。
"""

import logging
import math
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import os
import torch
from modelscope import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs.data import TokensPrompt

logger = logging.getLogger(__name__)


class MemoryReranker:
    """
    Memory reranker using Qwen3-Reranker model.
    使用Qwen3-Reranker模型对检索得到的memory进行重排序。
    """
    
    def __init__(
        self,
        model_path: str = "/mnt/cxzx/share/model_checkpoints/Qwen3-Reranker-4B",
        gpu_id: str = "2",
        max_model_len: int = 10000,
        gpu_memory_utilization: float = 0.8,
        tensor_parallel_size: int = 1
    ):
        """
        Initialize the memory reranker.
        
        Args:
            model_path: Path to the Qwen3-Reranker model
            gpu_id: GPU device ID to use
            max_model_len: Maximum model length
            gpu_memory_utilization: GPU memory utilization ratio
            tensor_parallel_size: Tensor parallel size for distributed inference
        """
        self.model_path = model_path
        self.gpu_id = gpu_id
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        
        # Set GPU environment
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        
        # Initialize model components
        self.tokenizer = None
        self.model = None
        self.suffix_tokens = None
        self.true_token = None
        self.false_token = None
        self.sampling_params = None
        
        # Default task instruction
        self.default_instruction = "Given a query, retrieve relevant information that answers the query"
        
        # Model state
        self._initialized = False
        
    def _initialize_model(self):
        """Initialize the reranker model and tokenizer."""
        if self._initialized:
            return
            
        try:
            logger.info(f"Initializing Qwen3-Reranker model from {self.model_path}")
            
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.tokenizer.padding_side = "left"
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Initialize model
            self.model = LLM(
                model=self.model_path,
                tensor_parallel_size=self.tensor_parallel_size,
                max_model_len=self.max_model_len,
                enable_prefix_caching=True,
                gpu_memory_utilization=self.gpu_memory_utilization
            )
            
            # Prepare suffix tokens and special tokens
            suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
            
            # Get true/false tokens for scoring
            self.true_token = self.tokenizer("yes", add_special_tokens=False).input_ids[0]
            self.false_token = self.tokenizer("no", add_special_tokens=False).input_ids[0]
            
            # Set sampling parameters
            self.sampling_params = SamplingParams(
                temperature=0,
                max_tokens=1,
                logprobs=20,
                allowed_token_ids=[self.true_token, self.false_token]
            )
            
            self._initialized = True
            logger.info("Qwen3-Reranker model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Qwen3-Reranker model: {e}")
            raise
    
    def _format_instruction(self, instruction: str, query: str, document: str) -> List[Dict[str, str]]:
        """
        Format the instruction for the reranker model.
        
        Args:
            instruction: Task instruction
            query: User query
            document: Document to evaluate
            
        Returns:
            Formatted instruction as list of role-content pairs
        """
        return [
            {
                "role": "system", 
                "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
            },
            {
                "role": "user", 
                "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {document}"
            }
        ]
    
    def _process_inputs(
        self, 
        pairs: List[Tuple[str, str]], 
        instruction: str, 
        max_length: int
    ) -> List[TokensPrompt]:
        """
        Process input pairs for the reranker model.
        
        Args:
            pairs: List of (query, document) pairs
            instruction: Task instruction
            max_length: Maximum token length
            
        Returns:
            List of processed token prompts
        """
        if not self._initialized:
            self._initialize_model()
            
        # Format messages for each pair
        messages = [
            self._format_instruction(instruction, query, doc) 
            for query, doc in pairs
        ]
        
        # Apply chat template
        tokenized_messages = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=False, 
            enable_thinking=False
        )
        
        # Truncate and add suffix tokens
        processed_messages = [
            msg[:max_length - len(self.suffix_tokens)] + self.suffix_tokens 
            for msg in tokenized_messages
        ]
        
        # Convert to TokensPrompt objects
        return [TokensPrompt(prompt_token_ids=msg) for msg in processed_messages]
    
    def _compute_scores(self, messages: List[TokensPrompt]) -> List[float]:
        """
        Compute relevance scores for the given messages.
        
        Args:
            messages: List of token prompts
            
        Returns:
            List of relevance scores (0-1, higher is better)
        """
        if not self._initialized:
            self._initialize_model()
            

        outputs = self.model.generate(messages, self.sampling_params, use_tqdm=False)
        scores = []
        
        for output in outputs:
            final_logits = output.outputs[0].logprobs[-1]
            
            # Get true and false logits
            true_logit = final_logits.get(self.true_token, -10)
            false_logit = final_logits.get(self.false_token, -10)
            
            # Convert logits to probabilities - handle Logprob objects
            if hasattr(true_logit, 'logprob'):
                true_logit = true_logit.logprob
            if hasattr(false_logit, 'logprob'):
                false_logit = false_logit.logprob
                
            # Convert logits to probabilities
            true_score = math.exp(true_logit)
            false_score = math.exp(false_logit)
            
            # Calculate final score
            score = true_score / (true_score + false_score)
            scores.append(score)
            
        return scores
        

    
    def rerank_memories(
        self,
        query: str,
        memories: List[str],
        instruction: Optional[str] = None,
        max_length: int = 8192
    ) -> List[Tuple[str, float]]:
        """
        Rerank memories based on relevance to the query.
        
        Args:
            query: User query
            memories: List of memory strings to rerank
            instruction: Custom instruction for reranking (optional)
            max_length: Maximum token length for processing
            
        Returns:
            List of (memory, score) tuples, sorted by score (descending)
        """
        if not memories:
            return []
            
        if not self._initialized:
            self._initialize_model()
        
       
        # Use default instruction if none provided
        if instruction is None:
            instruction = self.default_instruction
        
        # Create query-document pairs
        pairs = [(query, memory) for memory in memories]
        
        # Process inputs
        messages = self._process_inputs(
            pairs, 
            instruction, 
            max_length - len(self.suffix_tokens)
        )
        
        # Compute scores
        scores = self._compute_scores(messages)
        
        # Combine memories with scores
        memory_scores = list(zip(memories, scores))
        
        # Sort by score (descending)
        memory_scores.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Successfully reranked {len(memories)} memories")
        return memory_scores
 
    def rerank_memory_results(
        self,
        query: str,
        memory_results: List[Dict[str, Any]],
        instruction: Optional[str] = None,
        max_length: int = 8192
    ) -> List[Dict[str, Any]]:
        """
        Rerank memory results (with metadata) based on relevance to the query.
        
        Args:
            query: User query
            memory_results: List of memory result dictionaries
            instruction: Custom instruction for reranking (optional)
            max_length: Maximum token length for processing
            
        Returns:
            List of reranked memory results with added rerank_score
        """
        if not memory_results:
            return []
        
        # Extract memory contents
        memories = []
        for result in memory_results:
            if isinstance(result, dict):
                # Handle different result formats
                if 'content' in result:
                    memories.append(result['content'])
                elif 'item' in result and hasattr(result['item'], 'content'):
                    memories.append(result['item'].content)
                elif 'text' in result:
                    memories.append(result['text'])
                else:
                    # Fallback: convert to string
                    memories.append(str(result))
            else:
                memories.append(str(result))
        
        # Rerank memories
        reranked_memories = self.rerank_memories(query, memories, instruction, max_length)
        
        # Create reranked results with scores
        reranked_results = []
        for i, (memory, rerank_score) in enumerate(reranked_memories):
            # Find original result
            original_result = memory_results[i] if i < len(memory_results) else {}
            
            # Create new result with rerank score
            if isinstance(original_result, dict):
                new_result = original_result.copy()
                new_result['rerank_score'] = rerank_score
                new_result['rerank_rank'] = i + 1
            else:
                new_result = {
                    'content': memory,
                    'rerank_score': rerank_score,
                    'rerank_rank': i + 1
                }
            
            reranked_results.append(new_result)
        
        return reranked_results
    
    def batch_rerank(
        self,
        queries: List[str],
        memory_batches: List[List[str]],
        instruction: Optional[str] = None,
        max_length: int = 8192
    ) -> List[List[Tuple[str, float]]]:
        """
        Batch rerank multiple queries and memory sets.
        
        Args:
            queries: List of queries
            memory_batches: List of memory lists, one for each query
            instruction: Custom instruction for reranking (optional)
            max_length: Maximum token length for processing
            
        Returns:
            List of reranked memory results for each query
        """
        if len(queries) != len(memory_batches):
            raise ValueError("Number of queries must match number of memory batches")
        
        results = []
        for query, memories in zip(queries, memory_batches):
            reranked = self.rerank_memories(query, memories, instruction, max_length)
            results.append(reranked)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the reranker model."""
        return {
            "model_path": self.model_path,
            "gpu_id": self.gpu_id,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "tensor_parallel_size": self.tensor_parallel_size,
            "initialized": self._initialized,
            "default_instruction": self.default_instruction
        }
    
    def cleanup(self):
        """Clean up model resources."""
        if self._initialized and self.model:
            try:
                # Clean up VLLM model
                if hasattr(self.model, 'llm_engine'):
                    self.model.llm_engine.shutdown()
                self._initialized = False
                logger.info("Reranker model cleaned up successfully")
            except Exception as e:
                logger.error(f"Failed to cleanup reranker model: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()


# Convenience function for quick reranking
def rerank_memories(
    query: str,
    memories: List[str],
    model_path: str = "/mnt/cxzx/share/model_checkpoints/Qwen3-Reranker-4B",
    gpu_id: str = "6",
    instruction: Optional[str] = None
) -> List[Tuple[str, float]]:
    """
    Convenience function for quick memory reranking.
    
    Args:
        query: User query
        memories: List of memory strings to rerank
        model_path: Path to the reranker model
        gpu_id: GPU device ID
        instruction: Custom instruction for reranking
        
    Returns:
        List of (memory, score) tuples, sorted by score
    """
    reranker = MemoryReranker(model_path=model_path, gpu_id=gpu_id)
    try:
        return reranker.rerank_memories(query, memories, instruction)
    finally:
        reranker.cleanup()


# if __name__ == "__main__":
#     querys = ["What is the capital of China?"]
#     documents = [
#         "The capital of China is Shanghai.",
#     ]
#     reranker = MemoryReranker(model_path="/mnt/cxzx/share/model_checkpoints/Qwen3-Reranker-4B", gpu_id="2")
#     reranked = reranker.rerank_memories(querys, documents)
#     print(reranked)