"""
Simple Memory Extraction Base Class for EverMemOS

This module provides a simple base class for extracting memories
from boundary detection results (BoundaryResult).
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
import re, json, asyncio, uuid


# from ..prompts.zh.episode_mem_prompts import (
# from ..prompts.en.episode_mem_prompts import (
from ..prompts.eval.episode_mem_prompts import (
    EPISODE_GENERATION_PROMPT,
    GROUP_EPISODE_GENERATION_PROMPT,
    DEFAULT_CUSTOM_INSTRUCTIONS
)

from ..llm.llm_provider import LLMProvider

from .base_memory_extractor import MemoryExtractor, MemoryExtractRequest
from ..types import MemoryType, Memory, RawDataType, MemCell

from common_utils.datetime_utils import get_now_with_timezone

from core.observation.logger import get_logger
logger = get_logger(__name__)

@dataclass
class EpisodeMemory(Memory):
    """
    Simple result class for memory extraction.
    
    Contains the essential information for extracted memories.
    """
    event_id: str = field(default=None)
    
    def __post_init__(self):
        """Set memory_type to EPISODE_SUMMARY and call parent __post_init__."""
        self.memory_type = MemoryType.EPISODE_SUMMARY
        super().__post_init__()
    
@dataclass
class EpisodeMemoryExtractRequest(MemoryExtractRequest):
    pass


class EpisodeMemoryExtractor(MemoryExtractor):
    def __init__(self, llm_provider: LLMProvider | None = None):
        super().__init__(MemoryType.EPISODE_SUMMARY)
        self.llm_provider = llm_provider

    def _parse_timestamp(self, timestamp) -> datetime:
        """
        解析时间戳为 datetime 对象
        支持多种格式：数字时间戳、ISO格式字符串、数字字符串等
        """
        if isinstance(timestamp, datetime):
            return timestamp
        elif isinstance(timestamp, (int, float)):
            return datetime.fromtimestamp(timestamp)
        elif isinstance(timestamp, str):
            # Handle string timestamps (could be ISO format or timestamp string)
            try:
                if timestamp.isdigit():
                    return datetime.fromtimestamp(int(timestamp))
                else:
                    # Try parsing as ISO format
                    return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                # Fallback to current time if parsing fails
                logger.error(f"解析时间戳失败: {timestamp}")
                return get_now_with_timezone()
        else:
            # Unknown format, fallback to current time
            logger.error(f"解析时间戳失败: {timestamp}")
            return get_now_with_timezone()

    def _format_timestamp(self, dt: datetime) -> str:
        """
        格式化 datetime 为易读的字符串格式
        """
        weekday = dt.strftime("%A")  # Monday, Tuesday, etc.
        month_day = dt.strftime("%B %d, %Y")  # March 14, 2024
        time_of_day = dt.strftime("%I:%M %p")  # 3:00 PM
        return f"{month_day} ({weekday}) at {time_of_day} UTC"

    def get_conversation_text(self, data_list):
        lines = []
        for data in data_list:
            # Handle both RawData objects and dict objects
            if hasattr(data, 'content'):
                # RawData object
                speaker = data.content.get('speaker_name') or data.content.get('sender', 'Unknown')
                content = data.content['content']
                timestamp = data.content['timestamp']
            else:
                # Dict object
                speaker = data.get('speaker_name') or data.get('sender', 'Unknown')
                content = data['content']
                timestamp = data['timestamp']
            
            if timestamp:
                lines.append(f"[{timestamp}] {speaker}: {content}")
            else:
                lines.append(f"{speaker}: {content}")
        return "\n".join(lines)
    def get_conversation_json_text(self, data_list):
        lines = []
        for data in data_list:
            # Handle both RawData objects and dict objects
            if hasattr(data, 'content'):
                # RawData object
                speaker = data.content.get('speaker_name') or data.content.get('sender', 'Unknown')
                content = data.content['content']
                timestamp = data.content['timestamp']
            else:
                # Dict object
                speaker = data.get('speaker_name') or data.get('sender', 'Unknown')
                content = data['content']
                timestamp = data['timestamp']
            
            if timestamp:
                lines.append(
                f"""
                {{
                    "timestamp": {timestamp},
                    "speaker": {speaker},
                    "content": {content}
                }}""")
            else:
                lines.append(
                f"""
                {{
                    "speaker": {speaker},
                    "content": {content}
                }}""")
        return "\n".join(lines)
    def get_speaker_name_map(self, data_list: List[Dict[str, Any]]) -> Dict[str, str]:
        speaker_name_map = {}
        for data in data_list:
            if hasattr(data, 'content'):
                speaker_name_map[data.content.get('speaker_id')] = data.content.get('speaker_name')
            else:
                speaker_name_map[data.get('speaker_id')] = data.get('speaker_name')
        return speaker_name_map

    def _extract_participant_name_map(self, chat_raw_data_list: List[Dict[str, Any]]) -> List[str]:
        participant_name_map = {}
        for raw_data in chat_raw_data_list:
            if 'speaker_name' in raw_data and raw_data['speaker_name']:
                participant_name_map[raw_data['speaker_id']] = raw_data['speaker_name']
            if 'referList' in raw_data and raw_data['referList']:
                for refer_item in raw_data['referList']:
                    if isinstance(refer_item, dict):
                        if 'name' in refer_item and refer_item['_id']:
                            participant_name_map[refer_item['_id']] = refer_item['name']
        return participant_name_map

    async def extract_memory(self, request: EpisodeMemoryExtractRequest, use_group_prompt: bool = False) -> Optional[List[EpisodeMemory]] | Optional[MemCell]:
        logger.debug(f"📚 自动触发情景记忆提取...")
            
        if not request.memcell_list:
            return None
            
        # 获取第一个 memcell 来判断类型
        first_memcell = request.memcell_list[0]
        
        # 根据类型选择不同的处理方式
        if first_memcell.type == RawDataType.CONVERSATION:
            all_content_text = []
            prompt_template = ""
            # 对话类型处理
            for memcell in request.memcell_list:
                # conversation_text = self.get_conversation_text(memcell.original_data)
                conversation_text = self.get_conversation_json_text(memcell.original_data)
                all_content_text.append(conversation_text)
            
            # 根据使用场景选择提示词
            if use_group_prompt:
                # 与 extract_memcell 配套使用
                prompt_template = GROUP_EPISODE_GENERATION_PROMPT
                content_key = "conversation"
                time_key = "conversation_start_time"
            else:
                # 单独使用
                prompt_template = EPISODE_GENERATION_PROMPT
                content_key = "conversation"
                time_key = "conversation_start_time"
            default_title = "Conversation Episode"
        elif first_memcell.type == RawDataType.LINKDOC:
            # 文档类型处理 - 简化处理，不使用大模型
            start_time = first_memcell.timestamp  # 直接使用datetime对象
            start_time_str = self._format_timestamp(start_time)  # 只用于显示
            
            # 直接创建 EpisodeMemory，不调用大模型
            memroies = []
            if first_memcell.source_type == "memo":
                summary = f"Created memo: \n {first_memcell.summary}"
            else:
                summary = f"Received document from {first_memcell.source_type}: \n {first_memcell.summary}"
            memroies.extend([EpisodeMemory(
                memory_type=MemoryType.EPISODE_SUMMARY,
                user_id=user_id,
                ori_event_id_list=[memcell.event_id for memcell in request.memcell_list],
                timestamp=start_time,
                subject=first_memcell.subject or "Document Episode",  # 使用 memcell 的 subject
                summary=summary,  # 使用 memcell 的 summary
                keywords=first_memcell.keywords,
                episode=summary,  # episode 为空
                participants=first_memcell.participants,
                type=getattr(first_memcell, 'type', None),
                memcell_event_id_list=[memcell.event_id for memcell in request.memcell_list],
            ) for user_id in first_memcell.user_id_list])
            if first_memcell.participants:
                if first_memcell.source_type == "memo":
                    summary = f"Received shared memo: \n {first_memcell.summary}"
                else:
                    summary = f"Received document from {first_memcell.source_type}: \n {first_memcell.summary}"
                memroies.extend([EpisodeMemory(
                    memory_type=MemoryType.EPISODE_SUMMARY,
                    user_id=user_id,
                    ori_event_id_list=[memcell.event_id for memcell in request.memcell_list],
                    timestamp=start_time,
                    subject=first_memcell.subject or "Document Episode",  # 使用 memcell 的 subject
                    summary=summary,  # 使用 memcell 的 summary
                    keywords=first_memcell.keywords,
                    episode=summary,  # episode 为空
                    participants=first_memcell.participants,
                    type=getattr(first_memcell, 'type', None),
                    memcell_event_id_list=[memcell.event_id for memcell in request.memcell_list],
                ) for user_id in first_memcell.participants])
            return memroies
        elif first_memcell.type == RawDataType.EMAIL:
            # 邮件类型处理 - 简化处理，不使用大模型
            start_time = first_memcell.timestamp  # 直接使用datetime对象
            start_time_str = self._format_timestamp(start_time)  # 只用于显示
            
            # 直接创建 EpisodeMemory，不调用大模型
            memroies = []
            if first_memcell.email_type == "send":
                summary = f"Sent email: \n {first_memcell.summary}"
            else:
                summary = f"Received email: \n {first_memcell.summary}"
            memroies.extend([EpisodeMemory(
                memory_type=MemoryType.EPISODE_SUMMARY,
                user_id=user_id,
                ori_event_id_list=[memcell.event_id for memcell in request.memcell_list],
                timestamp=start_time,
                subject=first_memcell.subject or "Email Episode",  # 使用 memcell 的 subject
                summary=summary,  # 使用 memcell 的 summary
                keywords=first_memcell.keywords,
                episode=summary,  # episode 为空
                participants=first_memcell.participants,
                type=getattr(first_memcell, 'type', None),
                memcell_event_id_list=[memcell.event_id for memcell in request.memcell_list]
            ) for user_id in first_memcell.user_id_list])
            return memroies
        else: 
            pass


        # Extract earliest timestamp for context
        start_time = self._parse_timestamp(first_memcell.timestamp)
        start_time_str = self._format_timestamp(start_time)

        # Combine all content texts
        combined_content = "\n\n".join(all_content_text)

        # 构建 prompt
        if use_group_prompt:
            format_params = {
                time_key: start_time_str,
                content_key: combined_content,
                "custom_instructions": DEFAULT_CUSTOM_INSTRUCTIONS
            }
            prompt = prompt_template.format(**format_params)
            response = await self.llm_provider.generate(prompt)
            # 首先尝试提取代码块中的JSON
            if '```json' in response:
                # 提取代码块中的JSON内容
                start = response.find('```json') + 7
                end = response.find('```', start)
                if end > start:
                    json_str = response[start:end].strip()
                    data = json.loads(json_str)
                else:
                    # 尝试解析整个响应为JSON
                    data = json.loads(response)
            else:
                # 尝试匹配包含title和content的JSON对象
                json_match = re.search(r'\{[^{}]*"title"[^{}]*"content"[^{}]*\}', response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    # 尝试解析整个响应为JSON
                    data = json.loads(response)
            
            # Ensure we have required fields with fallback defaults
            if "title" not in data:
                data["title"] = default_title
            if "content" not in data:
                data["content"] = combined_content
            if "summary" not in data:
                # Generate a basic summary from content if not provided
                data["summary"] = data["content"]

            title = data["title"]
            content = data["content"]
            summary = data["summary"]

            # GROUP_EPISODE_GENERATION_PROMPT 模式：将情景记忆存储到 MemCell 中，返回 MemCell
            # 更新 MemCell 的 episode 字段
            for memcell in request.memcell_list:
                memcell.subject = title
                memcell.episode = content
            
            # 返回第一个 MemCell（已经包含了情景记忆内容）
            return first_memcell
        else:
            format_params = {
                time_key: start_time_str,
                content_key: combined_content,
                "custom_instructions": DEFAULT_CUSTOM_INSTRUCTIONS
            }
        
            participants = []
            [participants.extend(memcell.participants) for memcell in request.memcell_list]
            if not participants:
                participants = request.participants
            if not participants:
                participants = []
            
            all_memories = []
            if participants:
                all_original_data = []
                [all_original_data.extend(memcell.original_data) for memcell in request.memcell_list]
                participants_name_map = self.get_speaker_name_map(all_original_data)
                [participants_name_map.update(self._extract_participant_name_map(memcell.original_data)) for memcell in request.memcell_list]
                # 并发生成每个参与者的episode memory
                async def generate_memory_for_user(user_id: str, user_name: str) -> EpisodeMemory:
                     user_format_params = format_params.copy()
                     user_format_params["user_name"] = user_name
                     prompt = prompt_template.format(**user_format_params)
                     response = await self.llm_provider.generate(prompt)
                     
                     # 首先尝试提取代码块中的JSON
                     if '```json' in response:
                         # 提取代码块中的JSON内容
                         start = response.find('```json') + 7
                         end = response.find('```', start)
                         if end > start:
                             json_str = response[start:end].strip()
                             data = json.loads(json_str)
                         else:
                             # 尝试解析整个响应为JSON
                             data = json.loads(response)
                     else:
                         # 尝试匹配包含title和content的JSON对象
                         json_match = re.search(r'\{[^{}]*"title"[^{}]*"content"[^{}]*\}', response, re.DOTALL)
                         if json_match:
                             data = json.loads(json_match.group())
                         else:
                             # 尝试解析整个响应为JSON
                             data = json.loads(response)
                     
                     # Ensure we have required fields with fallback defaults
                     if "title" not in data:
                         data["title"] = default_title
                     if "content" not in data:
                         data["content"] = combined_content
                     if "summary" not in data:
                         # Generate a basic summary from content if not provided
                         data["summary"] = "\n".join([memcell.summary for memcell in request.memcell_list])
 
                     title = data["title"]
                     content = data["content"]
                     summary = data["summary"]
 
                     return EpisodeMemory(
                         memory_type=MemoryType.EPISODE_SUMMARY,
                         user_id=user_id,
                         ori_event_id_list=[memcell.event_id for memcell in request.memcell_list],
                         timestamp=start_time,
                         subject=title,
                         summary=summary,
                         episode=content,
                         group_id=request.group_id,
                         participants=participants,
                         type=getattr(first_memcell, 'type', None),
                         memcell_event_id_list=[memcell.event_id for memcell in request.memcell_list],
                     )
                 
                # 并发执行所有参与者的memory生成
                participant_memories = await asyncio.gather(
                    *[generate_memory_for_user(user_id, participants_name_map.get(user_id, user_id)) for user_id in participants],
                    return_exceptions=True
                )
                
                # 处理结果，过滤掉异常
                for memory in participant_memories:
                    if isinstance(memory, EpisodeMemory):
                        all_memories.append(memory)
                    else:
                        print(f"[EpisodicMemoryExtractor] Error generating memory: {memory}")
                
            for user_id in request.user_id_list:
                if user_id not in participants:
                    memory = EpisodeMemory(
                        memory_type=MemoryType.EPISODE_SUMMARY,
                        user_id=user_id,
                        ori_event_id_list=[memcell.event_id for memcell in request.memcell_list],
                        timestamp=start_time,
                        subject=title,
                        summary="\n".join([memcell.summary for memcell in request.memcell_list]),
                        episode="\n".join([memcell.episode for memcell in request.memcell_list]),
                        group_id=request.group_id,
                        participants=participants,
                        type=getattr(first_memcell, 'type', None),
                        memcell_event_id_list=[memcell.event_id for memcell in request.memcell_list],
                    )
                    all_memories.append(memory)
            return all_memories