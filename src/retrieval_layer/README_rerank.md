# MemoryReranker 使用说明

## 概述

`MemoryReranker` 是一个基于 Qwen3-Reranker 模型的memory重排序模块，用于对检索得到的memory进行重新排序，提高检索结果的相关性。

## 主要特性

- 基于 Qwen3-Reranker-0.6B 模型
- 支持自定义重排序指令
- 支持批量重排序
- 自动资源管理和清理
- 灵活的输入格式支持
- 异步操作支持

## 安装依赖

```bash
pip install vllm>=0.8.5
pip install modelscope
pip install torch
```

## 基本用法

### 1. 简单重排序

```python
from retrieval_layer.rerank import MemoryReranker

# 初始化重排序器
reranker = MemoryReranker(
    model_path="/mnt/cxzx/share/model_checkpoints/Qwen3-Reranker-0.6B",
    gpu_id="2"
)

# 定义查询和memory
query = "What is the capital of China?"
memories = [
    "The capital of China is Beijing.",
    "Shanghai is the largest city in China.",
    "China is located in East Asia."
]

# 执行重排序
reranked_results = reranker.rerank_memories(query, memories)

# 输出结果
for memory, score in reranked_results:
    print(f"Score: {score:.4f} | {memory}")

# 清理资源
reranker.cleanup()
```

### 2. 使用便利函数

```python
from retrieval_layer.rerank import rerank_memories

# 自动处理初始化和清理
reranked_results = rerank_memories(
    query="How to cook pasta?",
    memories=["Boil water", "Add salt", "Cook pasta"],
    gpu_id="2"
)
```

### 3. 自定义指令重排序

```python
reranker = MemoryReranker(gpu_id="2")

custom_instruction = "Given a query about cooking, retrieve relevant cooking instructions and tips."

reranked_results = reranker.rerank_memories(
    query="How to make coffee?",
    memories=["Grind beans", "Boil water", "Pour over"],
    instruction=custom_instruction
)
```

### 4. 重排序带元数据的memory结果

```python
memory_results = [
    {
        "content": "Exercise improves health.",
        "source": "medical_journal",
        "date": "2023-01-15"
    },
    {
        "content": "Regular exercise boosts mood.",
        "source": "psychology_study",
        "date": "2023-03-10"
    }
]

reranked_results = reranker.rerank_memory_results(
    query="What are the benefits of exercise?",
    memory_results=memory_results
)

# 结果包含重排序分数和排名
for result in reranked_results:
    print(f"Rank: {result['rerank_rank']}")
    print(f"Score: {result['rerank_score']}")
    print(f"Content: {result['content']}")
```

### 5. 批量重排序

```python
queries = ["What is ML?", "How to cook?", "Weather info?"]
memory_batches = [
    ["ML is AI subset", "Coffee beans", "ML uses algorithms"],
    ["Grind beans", "ML needs data", "Boil water"],
    ["Check forecast", "ML models train", "Look at sky"]
]

batch_results = reranker.batch_rerank(queries, memory_batches)
```

## 配置参数

### MemoryReranker 初始化参数

- `model_path`: Qwen3-Reranker模型路径
- `gpu_id`: 使用的GPU设备ID
- `max_model_len`: 最大模型长度 (默认: 10000)
- `gpu_memory_utilization`: GPU内存使用率 (默认: 0.8)
- `tensor_parallel_size`: 张量并行大小 (默认: 1)

### 重排序参数

- `instruction`: 自定义重排序指令
- `max_length`: 最大token长度 (默认: 8192)

## 输出格式

### rerank_memories 输出

返回 `List[Tuple[str, float]]`，每个元素包含：
- `memory`: memory内容字符串
- `score`: 相关性分数 (0-1，越高越相关)

### rerank_memory_results 输出

返回 `List[Dict[str, Any]]`，每个元素包含：
- 原始结果的所有字段
- `rerank_score`: 重排序分数
- `rerank_rank`: 重排序后的排名

## 性能优化建议

1. **批量处理**: 使用 `batch_rerank` 进行批量重排序
2. **GPU内存**: 根据GPU内存调整 `gpu_memory_utilization`
3. **模型长度**: 根据实际需求调整 `max_model_len`
4. **资源管理**: 及时调用 `cleanup()` 释放GPU资源

## 错误处理

- 模型初始化失败时会抛出异常
- 推理失败时会返回默认分数 (0.5)
- 自动记录错误日志

## 注意事项

1. 确保GPU设备可用且有足够内存
2. 模型路径必须存在且可访问
3. 输入memory不应为空
4. 及时清理资源避免GPU内存泄漏

## 完整示例

查看 `rerank_example.py` 文件获取更多使用示例。

## 与现有系统集成

`MemoryReranker` 可以与现有的检索系统无缝集成：

```python
from retrieval_layer.retriever import UnifiedRetriever
from retrieval_layer.rerank import MemoryReranker

# 检索memory
retriever = UnifiedRetriever(method="nemori", data_base="memory_base")
memories = await retriever.retrieve("What is AI?", top_k=20)

# 重排序
reranker = MemoryReranker(gpu_id="2")
reranked_memories = reranker.rerank_memories("What is AI?", memories)

# 使用重排序结果
top_memories = [memory for memory, score in reranked_memories[:5]]
``` 