"""
记忆检索服务

此模块提供基于检索方法的记忆检索服务，支持多种检索方法如关键词检索、向量检索等。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Any, Optional

from core.di import get_bean, service

logger = logging.getLogger(__name__)


class RetrieveMemServiceInterface(ABC):
    """记忆检索服务接口"""
    
    @abstractmethod
    async def retrieve(self, query: str, method: str = "keyword", top_k: int = 10, user_id: str = "default", 
                      memory_types: List[str] = None, start_time: Optional[float] = None, 
                      end_time: Optional[float] = None) -> List[str]:
        """
        根据查询字符串检索记忆
        
        Args:
            query: 查询字符串
            method: 检索方法 ("keyword", "vector", "mix")
            top_k: 返回结果数量
            user_id: 用户ID
            memory_types: 记忆类型列表
            start_time: 开始时间戳（Unix时间戳）
            end_time: 结束时间戳（Unix时间戳）
            
        Returns:
            检索到的记忆内容列表
        """
        pass


@service(name="retrieve_mem_service", primary=True)
class RetrieveMemServiceImpl(RetrieveMemServiceInterface):
    """记忆检索服务实现"""
    
    def __init__(self):
        """初始化检索服务"""
        self._retrievers = {}  # 缓存不同方法的检索器，key为 (method, user_id)
        self._initialized_methods = set()  # 已初始化的方法集合，存储 (method, user_id) 元组
        logger.info("RetrieveMemServiceImpl initialized")
    
    async def _get_retriever(self, method: str, data_base: str, user_id: str = None):
        """获取或初始化指定方法的检索器（懒加载）"""
        cache_key = (method, user_id)
        
        if cache_key not in self._initialized_methods:
            try:
                from retrieval_layer.retriever import UnifiedRetriever
                
                # 创建 UnifiedRetriever 实例
                retriever = UnifiedRetriever(
                    version="episode_summary", #version主要永用于定位索引和数据库具体位置
                    method=method,
                    data_base=data_base,
                    top_k=5,
                    user_id=user_id,
                )
                
                # 初始化检索器
                await retriever.initialize()
                
                # 缓存检索器
                self._retrievers[cache_key] = retriever
                self._initialized_methods.add(cache_key)
                
                logger.debug(f"Retriever for method '{method}' and user_id '{user_id}' initialized and cached")
                
            except Exception as e:
                logger.error(f"Failed to initialize retriever for method '{method}': {e}")
                raise
        
        return self._retrievers[cache_key]
    
    async def retrieve(self, query: str, method: str = "keyword", top_k: int = 10, user_id: str = "default",
                      memory_types: List[str] = None, start_time: Optional[float] = None, 
                      end_time: Optional[float] = None) -> List[str]:
        """
        根据查询字符串检索记忆
        
        Args:
            query: 查询字符串
            method: 检索方法
            top_k: 返回结果数量
            user_id: 用户ID
            memory_types: 记忆类型列表
            start_time: 开始时间戳（Unix时间戳）
            end_time: 结束时间戳（Unix时间戳）
            
        Returns:
            检索到的记忆内容列表
        """
        try:
            # 设置数据基础路径
            data_base = "/home/gongjie/b001-memsys/memory_base/dynamic_memory_base/storages/"
            
            # 获取或初始化检索器
            retriever = await self._get_retriever(method, data_base, user_id)
            
            # 执行检索
            results = await retriever.retrieve(
                query=query,
                top_k=top_k,
                owner_id=user_id
            )
            
            logger.debug(f"RetrieveMemService returned {len(results)} results for method '{method}' and query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Error in retrieve_mem with method '{method}': {e}")
            return []


def get_retrieve_mem_service() -> RetrieveMemServiceInterface:
    """获取记忆检索服务实例
    
    通过依赖注入框架获取服务实例，支持单例模式。
    """
    return get_bean("retrieve_mem_service")


# 便捷函数
async def retrieve_memories(query: str, method: str = "keyword", top_k: int = 40, user_id: str = "default") -> List[str]:
    """
    便捷函数：根据查询字符串检索记忆
    
    Args:
        query: 查询字符串
        method: 检索方法
        top_k: 返回结果数量
        user_id: 用户ID
        
    Returns:
        检索到的记忆内容列表
    """
    service = get_retrieve_mem_service()
    return await service.retrieve(query, method, top_k, user_id)


