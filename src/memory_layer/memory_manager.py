from dataclasses import dataclass
import datetime
import time

from core.observation.logger import get_logger

from .llm.llm_provider import LLMProvider
from .memcell_extractor.conv_memcell_extractor import ConvMemCellExtractor
from .memcell_extractor.base_memcell_extractor import RawData
from .memcell_extractor.conv_memcell_extractor import ConversationMemCellExtractRequest
from .types import MemCell
from .memory_extractor.episode_memory_extractor import EpisodeMemoryExtractor, EpisodeMemoryExtractRequest, Memory
from .memory_extractor.profile_memory_extractor import ProfileMemoryExtractor, ProfileMemoryExtractRequest
from .memory_extractor.group_profile_memory_extractor import GroupProfileMemoryExtractor, GroupProfileMemoryExtractRequest
import os
from .types import RawDataType, MemoryType
from .memcell_extractor.base_memcell_extractor import StatusResult
from typing import List, Optional


@dataclass
class MemorizeRequest:
    history_raw_data_list: list[RawData]
    new_raw_data_list: list[RawData]
    raw_data_type: RawDataType
    # 整个group全量的user_id列表
    user_id_list: List[str]
    group_id: Optional[str] = None
    group_name: Optional[str] = None
    current_time: Optional[datetime] = None

@dataclass
class MemorizeOfflineRequest:
    memorize_from: datetime
    memorize_to: datetime
    
class MemoryManager:
    def __init__(self):
        # Conversation MemCell LLM Provider - 从环境变量读取配置
        self.conv_memcall_llm_provider = LLMProvider(
            provider_type=os.getenv("CONV_MEMCELL_LLM_PROVIDER", "openai"),
            model=os.getenv("CONV_MEMCELL_LLM_MODEL", "Qwen3-235B"),
            base_url=os.getenv("CONV_MEMCELL_LLM_BASE_URL", "http://180.184.148.131:30080/v1"),
            api_key=os.getenv("CONV_MEMCELL_LLM_API_KEY", "123"),
            temperature=float(os.getenv("CONV_MEMCELL_LLM_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("CONV_MEMCELL_LLM_MAX_TOKENS", "16384"))
        )
        
        # Episode Memory Extractor LLM Provider - 从环境变量读取配置  
        self.episode_memory_extractor_llm_provider = LLMProvider(
            provider_type=os.getenv("EPISODE_MEMORY_LLM_PROVIDER", "openai"),
            model=os.getenv("EPISODE_MEMORY_LLM_MODEL", "Qwen3-235B"),
            base_url=os.getenv("EPISODE_MEMORY_LLM_BASE_URL", "http://180.184.148.131:30080/v1"),
            api_key=os.getenv("EPISODE_MEMORY_LLM_API_KEY", "123"),
            temperature=float(os.getenv("EPISODE_MEMORY_LLM_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("EPISODE_MEMORY_LLM_MAX_TOKENS", "16384"))
        )
        
        # Profile Memory Extractor LLM Provider - 从环境变量读取配置
        self.profile_memory_extractor_llm_provider = LLMProvider(
            provider_type=os.getenv("PROFILE_MEMORY_LLM_PROVIDER", "openai"),
            model=os.getenv("PROFILE_MEMORY_LLM_MODEL", "Qwen3-235B"),
            base_url=os.getenv("PROFILE_MEMORY_LLM_BASE_URL", "http://180.184.148.131:30080/v1"),
            api_key=os.getenv("PROFILE_MEMORY_LLM_API_KEY", "123"),
            temperature=float(os.getenv("PROFILE_MEMORY_LLM_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("PROFILE_MEMORY_LLM_MAX_TOKENS", "16384"))
        )
        
        # LinkDoc MemCell LLM Provider - 从环境变量读取配置
        self.linkdoc_memcall_llm_provider = LLMProvider(
            provider_type=os.getenv("LINKDOC_MEMCELL_LLM_PROVIDER", "openai"),
            model=os.getenv("LINKDOC_MEMCELL_LLM_MODEL", "Qwen3-235B"),
            base_url=os.getenv("LINKDOC_MEMCELL_LLM_BASE_URL", "http://180.184.148.131:30080/v1"),
            api_key=os.getenv("LINKDOC_MEMCELL_LLM_API_KEY", "123"),
            temperature=float(os.getenv("LINKDOC_MEMCELL_LLM_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("LINKDOC_MEMCELL_LLM_MAX_TOKENS", "16384"))
        )
        
        # Email MemCell LLM Provider - 从环境变量读取配置
        self.email_memcall_llm_provider = LLMProvider(
            provider_type=os.getenv("EMAIL_MEMCELL_LLM_PROVIDER", "openai"),
            model=os.getenv("EMAIL_MEMCELL_LLM_MODEL", "Qwen3-235B"),
            base_url=os.getenv("EMAIL_MEMCELL_LLM_BASE_URL", "http://180.184.148.131:30080/v1"),
            api_key=os.getenv("EMAIL_MEMCELL_LLM_API_KEY", "123"),
            temperature=float(os.getenv("EMAIL_MEMCELL_LLM_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("EMAIL_MEMCELL_LLM_MAX_TOKENS", "16384"))
        )


        
    async def extract_memcell(self, history_raw_data_list: list[RawData], new_raw_data_list: list[RawData], raw_data_type: RawDataType, group_id: Optional[str] = None, group_name: Optional[str] = None, user_id_list: Optional[List[str]] = None, old_memory_list: Optional[List[Memory]] = None) -> tuple[Optional[MemCell], Optional[StatusResult]]:
        logger = get_logger(__name__)
        extractor = None
        now = time.time()
        request = ConversationMemCellExtractRequest(history_raw_data_list, new_raw_data_list, user_id_list=user_id_list, group_id=group_id, group_name=group_name, old_memory_list=old_memory_list)
        extractor = ConvMemCellExtractor(self.conv_memcall_llm_provider)
        result = await extractor.extract_memcell(request)
        logger.debug(f"提取MemCell完成, raw_data_type: {raw_data_type}, 耗时: {time.time() - now}秒")
        return result
 
    async def extract_memory(self, memcell_list: list[MemCell], memory_type: MemoryType, user_ids: List[str], group_id: Optional[str] = None, group_name: Optional[str] = None, old_memory_list: Optional[List[Memory]] = None, user_organization: Optional[List] = None) -> List[Memory]:
        extractor = None
        request = None
        
        if memory_type == MemoryType.EPISODE_SUMMARY:
            extractor = EpisodeMemoryExtractor(self.episode_memory_extractor_llm_provider)
            request = EpisodeMemoryExtractRequest(
                memcell_list=memcell_list,
                user_id_list=user_ids,
                group_id=group_id,
                old_memory_list=old_memory_list
            )
        elif memory_type == MemoryType.PROFILE:
            if memcell_list[0].type == RawDataType.CONVERSATION:
                extractor = ProfileMemoryExtractor(self.profile_memory_extractor_llm_provider)
                request = ProfileMemoryExtractRequest(
                    memcell_list=memcell_list,
                    user_id_list=user_ids,
                    group_id=group_id,
                    old_memory_list=old_memory_list
                )
        elif memory_type == MemoryType.GROUP_PROFILE:
            extractor = GroupProfileMemoryExtractor(self.profile_memory_extractor_llm_provider)
            request = GroupProfileMemoryExtractRequest(
                memcell_list=memcell_list,
                user_id_list=user_ids,
                group_id=group_id,
                group_name=group_name,
                old_memory_list=old_memory_list,
                user_organization=None
            )
            
        if extractor == None or request == None:
            return []
        return await extractor.extract_memory(request)
        
    