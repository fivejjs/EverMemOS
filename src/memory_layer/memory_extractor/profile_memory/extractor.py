"""Profile memory extractor implementation."""

from __future__ import annotations

import asyncio
import ast
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from core.observation.logger import get_logger

from ...llm.llm_provider import LLMProvider
from ...prompts.en.profile_mem_part1_prompts import CONVERSATION_PROFILE_PART1_EXTRACTION_PROMPT
from ...prompts.en.profile_mem_part2_prompts import CONVERSATION_PROFILE_PART2_EXTRACTION_PROMPT
from ...prompts.en.profile_mem_part3_prompts import CONVERSATION_PROFILE_PART3_EXTRACTION_PROMPT
from ...types import MemoryType, MemCell
from .conversation import (
    annotate_relative_dates,
    build_conversation_text,
    build_episode_text,
    extract_group_important_info,
    extract_user_mapping_from_memcells,
    is_important_to_user,
    merge_group_importance_evidence,
)
from .data_normalize import (
    accumulate_old_memory_entry,
    convert_projects_to_dataclass,
    merge_profiles,
    merge_projects_participated,
    merge_single_profile,
    profile_payload_to_memory,
    project_to_dict,
    remove_evidences_from_profile,
)
from .merger import convert_important_info_to_evidence
from .types import GroupImportanceEvidence, ImportanceEvidence, ProfileMemory, ProfileMemoryExtractRequest, ProjectInfo
from ..base_memory_extractor import MemoryExtractor

logger = get_logger(__name__)


class ProfileMemoryExtractor(MemoryExtractor):
    """Extractor for user profile information from conversations."""

    _default_evidence_date: Optional[str] = None
    _conversation_date_map: Dict[str, str] = {}

    def __init__(self, llm_provider: LLMProvider | None = None):
        super().__init__(MemoryType.PROFILE)
        self.llm_provider = llm_provider

    async def extract_memory(
        self,
        request: ProfileMemoryExtractRequest,
    ) -> Optional[List[ProfileMemory]]:
        """Extract profile memories from conversation memcells."""
        if not request.memcell_list:
            return None

        self.__class__._default_evidence_date = None
        self.__class__._conversation_date_map = {}

        # Extract complete user_id to user_name mapping from all memcells once
        user_id_to_name = extract_user_mapping_from_memcells(request.memcell_list)

        conversation_date_map = self.__class__._conversation_date_map
        latest_date: Optional[str] = None
        all_conversation_text: List[str] = []
        all_episode_text: List[str] = []
        valid_conversation_ids: Set[str] = set()
        for memcell in request.memcell_list:
            conversation_text, conversation_id = build_conversation_text(memcell, user_id_to_name)
            all_conversation_text.append(conversation_text)

            episode_text, episode_id = build_episode_text(memcell, user_id_to_name)
            all_episode_text.append(episode_text)

            timestamp_value = getattr(memcell, "timestamp", None)
            dt_value = self._parse_timestamp(timestamp_value)
            if dt_value is None:
                msg_timestamp = self._extract_first_message_timestamp(memcell)
                if msg_timestamp is not None:
                    dt_value = self._parse_timestamp(msg_timestamp)
            date_str: Optional[str] = None
            if dt_value:
                date_str = dt_value.date().isoformat()
                latest_date = date_str

            if conversation_id:
                valid_conversation_ids.add(conversation_id)
                if date_str:
                    conversation_date_map.setdefault(conversation_id, date_str)
            event_id = getattr(memcell, "event_id", None)
            if event_id:
                event_id_str = str(event_id)
                valid_conversation_ids.add(event_id_str)
                if date_str:
                    conversation_date_map.setdefault(event_id_str, date_str)

        self.__class__._default_evidence_date = latest_date
        default_date = self.__class__._default_evidence_date

        resolved_group_id = request.group_id
        if not resolved_group_id:
            for memcell in request.memcell_list:
                candidate_group_id = getattr(memcell, "group_id", None)
                if candidate_group_id:
                    resolved_group_id = candidate_group_id
                    break
        resolved_group_id = resolved_group_id or ""

        participants_profile_list: List[Dict[str, Any]] = []
        participants_profile_list_no_evidences: List[Dict[str, Any]] = []
        participants_base_memory_map: Dict[str, Dict[str, Any]] = {}

        if request.old_memory_list:
            for mem in request.old_memory_list:
                if mem.memory_type == MemoryType.PROFILE:
                    accumulate_old_memory_entry(mem, participants_profile_list)
                    if participants_profile_list:
                        profile_obj_no_evidences = remove_evidences_from_profile(
                            participants_profile_list[-1]
                        )
                        participants_profile_list_no_evidences.append(profile_obj_no_evidences)
                elif mem.memory_type == MemoryType.BASE_MEMORY:
                    base_memory_obj: Dict[str, Any] = {"user_id": mem.user_id}

                    if getattr(mem, "position", None):
                        base_memory_obj["position"] = mem.position
                    if getattr(mem, "base_location", None):
                        base_memory_obj["base_location"] = mem.base_location
                    if getattr(mem, "department", None):
                        base_memory_obj["department"] = mem.department

                    if len(base_memory_obj) > 1:
                        participants_base_memory_map[mem.user_id] = base_memory_obj

        # Build两个提示词
        prompt_part1 = self._build_profile_prompt(
            CONVERSATION_PROFILE_PART1_EXTRACTION_PROMPT,
            all_conversation_text,
            participants_profile_list_no_evidences,
            participants_base_memory_map,
            request,
        )
        prompt_part2 = self._build_profile_prompt(
            CONVERSATION_PROFILE_PART2_EXTRACTION_PROMPT,
            all_conversation_text,
            participants_profile_list_no_evidences,
            participants_base_memory_map,
            request,
        )

        # 定义异步调用函数
        async def invoke_llm(
            prompt: str,
            part_label: str,
        ) -> Optional[List[Dict[str, Any]]]:
            try:
                response_text = await self.llm_provider.generate(prompt, temperature=0.3)
            except Exception as exc:
                logger.error(
                    f"[ProfileMemoryExtractor] {part_label} profile generation error: {exc}"
                )
                return None

            annotated_response = self._annotate_relative_dates(response_text)
            return self._extract_user_profiles_from_response(
                annotated_response,
                part_label,
            )

        # 并发调用两个LLM
        # task_part1 = asyncio.create_task(
        #     invoke_llm(prompt_part1, "personal profile part"),
        # )
        # task_part2 = asyncio.create_task(
        #     invoke_llm(prompt_part2, "project profile part"),
        # )
        #
        # profiles_part1, profiles_part2 = await asyncio.gather(task_part1, task_part2)

        # 串行调用两个LLM
        profiles_part1 = await invoke_llm(prompt_part1, "personal profile part")
        profiles_part2 = await invoke_llm(prompt_part2, "project profile part")

        # 合并结果
        if not profiles_part1 and not profiles_part2:
            logger.warning("[ProfileMemoryExtractor] Both parts returned no profiles")
            return None

        # Use pre-extracted user_id_to_name mapping for validation
        participant_user_ids: Set[str] = set(user_id_to_name.keys())

        part1_map: Dict[str, Dict[str, Any]] = {}
        if profiles_part1:
            for profile in profiles_part1:
                if not isinstance(profile, dict):
                    continue
                user_id = str(profile.get("user_id", "")).strip()
                if not user_id:
                    logger.error(
                        "[ProfileMemoryExtractor] LLM returned empty user_id in part1; skipping profile"
                    )
                    continue
                # Validate user_id against participants_profile_list
                if participant_user_ids and user_id not in participant_user_ids:
                    logger.error(
                        "[ProfileMemoryExtractor] LLM returned user_id %s not found in participants_profile_list; skipping profile",
                        user_id,
                    )
                    continue
                part1_map[user_id] = profile

        part2_map: Dict[str, Dict[str, Any]] = {}
        if profiles_part2:
            for profile in profiles_part2:
                if not isinstance(profile, dict):
                    continue
                user_id = str(profile.get("user_id", "")).strip()
                if not user_id:
                    logger.error(
                        "[ProfileMemoryExtractor] LLM returned empty user_id in part2; skipping profile"
                    )
                    continue
                # Validate user_id against participants_profile_list
                if participant_user_ids and user_id not in participant_user_ids:
                    logger.error(
                        "[ProfileMemoryExtractor] LLM returned user_id %s not found in participants_profile_list; skipping profile",
                        user_id,
                    )
                    continue
                part2_map[user_id] = profile

        # 合并两部分的数据
        combined_user_ids = set(part1_map) | set(part2_map)
        if not combined_user_ids:
            logger.warning("[ProfileMemoryExtractor] No valid user_ids found in combined results")
            return None

        user_profiles_data: List[Dict[str, Any]] = []
        for user_id in combined_user_ids:
            combined_profile: Dict[str, Any] = {"user_id": user_id}

            # 合并part1的数据（个人属性：opinion_tendency, working_habit_preference, soft_skills, personality, way_of_decision_making,hard_skills）
            if user_id in part1_map:
                part1_profile = part1_map[user_id]
                for key, value in part1_profile.items():
                    if key != "user_id" and value:
                        combined_profile[key] = value

            # 合并part2的数据（role_responsibility + hard_skills + 项目经历）
            if user_id in part2_map:
                part2_profile = part2_map[user_id]
                # user_name优先使用part1，如果没有则使用part2
                if "user_name" not in combined_profile and "user_name" in part2_profile:
                    combined_profile["user_name"] = part2_profile["user_name"]
                # 从part2提取role_responsibility和hard_skills
                if "role_responsibility" in part2_profile:
                    combined_profile["role_responsibility"] = part2_profile["role_responsibility"]
                if "hard_skills" in part2_profile:
                    combined_profile["hard_skills"] = part2_profile["hard_skills"]
                # projects_participated只在part2中
                if "projects_participated" in part2_profile:
                    combined_profile["projects_participated"] = part2_profile["projects_participated"]

            user_profiles_data.append(combined_profile)

        profile_memories: List[ProfileMemory] = []
        for profile_data in user_profiles_data:
            if not isinstance(profile_data, dict):
                continue
            profile_user_id = str(profile_data.get("user_id") or "").strip()

            projects_participated = profile_data.get("projects_participated")
            project_payload_override: Optional[Dict[str, Any]] = None
            if isinstance(projects_participated, list):
                project_infos = convert_projects_to_dataclass(
                    projects_participated,
                    valid_conversation_ids=valid_conversation_ids,
                    default_date=default_date,
                    conversation_date_map=conversation_date_map,
                )
                if len(project_infos) > 1:
                    logger.error(
                        "[ProfileMemoryExtractor] Unexpected multiple projects for user %s in group %s",
                        profile_data.get("user_id"),
                        request.group_id,
                    )
                    project_infos = merge_projects_participated(None, project_infos)
                if project_infos:
                    project_payload_override = {"projects_participated": project_infos}

            profile_memory = profile_payload_to_memory(
                profile_data,
                group_id=resolved_group_id,
                project_data=project_payload_override,
                valid_conversation_ids=valid_conversation_ids,
                default_date=default_date,
                conversation_date_map=conversation_date_map,
            )
            if profile_memory:
                profile_memories.append(profile_memory)
        
        merged_profiles = merge_profiles(
            profile_memories,
            participants_profile_list,
            group_id=resolved_group_id,
            default_date=default_date,
            valid_conversation_ids=valid_conversation_ids,
            conversation_date_map=conversation_date_map,
        )

        # Debug logging for specific user and group
        for profile in merged_profiles:
            if profile.user_id == "5fc48d31db1c7c1701161e34" and profile.group_id == "676a2d05d4dd26352df7fad3":
                logger.debug(
                    "[ProfileMemoryExtractor] Debug:  profile=%s",
                    profile
                )

        important_info = extract_group_important_info(request.memcell_list, request.group_id)
        new_evidence_list = convert_important_info_to_evidence(important_info)
        for profile in merged_profiles:
            old_evidence: Optional[GroupImportanceEvidence] = profile.group_importance_evidence
            new_evidence = merge_group_importance_evidence(
                old_evidence,
                new_evidence_list,
                user_id=profile.user_id,
            )
            if new_evidence:
                new_evidence.is_important = is_important_to_user(new_evidence.evidence_list)
                profile.group_importance_evidence = new_evidence

        return merged_profiles

    def _build_profile_prompt(
        self,
        prompt_template: str,
        all_conversation_text: List[str],
        participants_profile_list_no_evidences: List[Dict[str, Any]],
        participants_base_memory_map: Dict[str, Dict[str, Any]],
        request: ProfileMemoryExtractRequest,
    ) -> str:
        """Build prompt from template with variable replacements."""
        return (
            prompt_template.replace(
                "{conversation}",
                "\n".join(all_conversation_text),
            )
            .replace(
                "{participants_profile}",
                json.dumps(participants_profile_list_no_evidences, ensure_ascii=False),
            )
            .replace(
                "{participants_baseMemory}",
                json.dumps(participants_base_memory_map, ensure_ascii=False),
            )
            .replace("{project_name}", (request.group_name or ""))
            .replace("{project_id}", request.group_id or "")
        )

    def _annotate_relative_dates(self, response: str) -> str:
        """Annotate relative dates in response."""
        return annotate_relative_dates(response, default_date=self._default_evidence_date)

    def _extract_user_profiles_from_response(
        self,
        response: str,
        part_label: str,
    ) -> Optional[List[Dict[str, Any]]]:
        """Extract user profiles from LLM response."""
        try:
            data = self._parse_profile_response_payload(response)
            user_profiles = data.get("user_profiles", [])
            if not user_profiles:
                logger.warning(
                    f"[ProfileMemoryExtractor] No user profiles found in {part_label}"
                )
                return None
            return user_profiles
        except Exception as exc:
            logger.error(
                f"[ProfileMemoryExtractor] Failed to parse {part_label} response: {exc}"
            )
            if response:
                logger.error(
                    f"[ProfileMemoryExtractor] {part_label} response preview: {response[:500]}"
                )
            return None

    @staticmethod
    def _parse_timestamp(timestamp: Any) -> Optional[datetime]:
        if isinstance(timestamp, datetime):
            return timestamp
        if isinstance(timestamp, (int, float)):
            return datetime.fromtimestamp(timestamp)
        if isinstance(timestamp, str):
            ts_value = timestamp.strip()
            iso_timestamp = ts_value.replace("Z", "+00:00") if ts_value.endswith("Z") else ts_value
            try:
                return datetime.fromisoformat(iso_timestamp)
            except ValueError:
                return None
        return None

    @staticmethod
    def _extract_first_message_timestamp(memcell: MemCell) -> Optional[Any]:
        """Return the first available timestamp from a memcell's original data."""
        for message in getattr(memcell, "original_data", []) or []:
            if hasattr(message, "content"):
                ts_value = message.content.get("timestamp")
            else:
                ts_value = message.get("timestamp")
            if ts_value:
                return ts_value
        return None

    @staticmethod
    def _parse_profile_response_payload(response: str) -> Dict[str, Any]:
        """Best-effort JSON extraction from LLM responses with optional markdown fences."""
        if not response:
            raise ValueError("empty response")

        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))

        parsed = ast.literal_eval(response)
        if isinstance(parsed, (dict, list)):
            return parsed
        return json.loads(parsed)

    @classmethod
    def _profile_payload_to_memory(
        cls,
        profile_data: Dict[str, Any],
        group_id: str,
        project_data: Optional[Dict[str, Any]] = None,
        valid_conversation_ids: Optional[Set[str]] = None,
    ) -> Optional[ProfileMemory]:
        """Compatibility wrapper for legacy unit tests."""
        return profile_payload_to_memory(
            profile_data,
            group_id=group_id,
            project_data=project_data,
            valid_conversation_ids=valid_conversation_ids,
            default_date=cls._default_evidence_date,
            conversation_date_map=cls._conversation_date_map,
        )

    @classmethod
    def _project_to_dict(cls, project: Any) -> Dict[str, Any]:
        """Compatibility wrapper for project serialization."""
        return project_to_dict(project)

    @classmethod
    def _merge_single_profile(
        cls,
        existing: ProfileMemory,
        new: ProfileMemory,
        group_id: str,
    ) -> ProfileMemory:
        """Compatibility wrapper delegating to the shared merge helper."""
        return merge_single_profile(existing, new, group_id=group_id)

    @classmethod
    def _convert_projects_to_dataclass(
        cls,
        projects_data: List[Any],
    ) -> List[ProjectInfo]:
        """Compatibility wrapper for legacy access to project conversion."""
        return convert_projects_to_dataclass(
            projects_data,
            default_date=cls._default_evidence_date,
            conversation_date_map=cls._conversation_date_map,
        )

    @classmethod
    def _merge_projects_participated(
        cls,
        existing_projects: Optional[List[ProjectInfo]],
        incoming_projects: Optional[List[ProjectInfo]],
    ) -> List[ProjectInfo]:
        """Compatibility wrapper around project merge helper."""
        return merge_projects_participated(existing_projects, incoming_projects)

    @classmethod
    def _extract_group_important_info(
        cls,
        memcells: List[MemCell],
        group_id: str,
    ) -> Dict[str, Any]:
        """Compatibility wrapper for legacy tests."""
        return extract_group_important_info(memcells, group_id)

    @classmethod
    def _convert_important_info_to_evidence(
        cls,
        important_info: Dict[str, Any],
    ) -> List[ImportanceEvidence]:
        """Compatibility wrapper for legacy tests."""
        return convert_important_info_to_evidence(important_info)

    async def extract_profile_companion(
        self,
        request: ProfileMemoryExtractRequest,
    ) -> Optional[List[ProfileMemory]]:
        """Extract companion profile memories using Part3 prompts (90 personality dimensions).
        
        This function analyzes conversation memcells to extract comprehensive personality profiles
        based on 90 dimensions including psychological traits, AI alignment preferences,
        and content platform interests.
        
        Args:
            request: ProfileMemoryExtractRequest containing memcells and optional old memories
            
        Returns:
            Optional[List[ProfileMemory]]: List of extracted profile memories with 90-dimension analysis,
                                           or None if extraction failed
        """
        if not request.memcell_list:
            logger.warning("[ProfileMemoryExtractor] No memcells provided for companion extraction")
            return None
        
        # Extract user mapping from memcells and build conversation text
        user_id_to_name = extract_user_mapping_from_memcells(request.memcell_list)
        # print(f"[ProfileMemoryExtractor] user_id_to_name: {user_id_to_name}")
        # Build conversation text from all memcells
        conversation_lines: List[str] = []
        user_profiles: Dict[str, Dict[str, Any]] = {}  # user_id -> {name, message_count}
        
        for memcell in request.memcell_list:
            conversation_text, conversation_id = build_conversation_text(memcell, user_id_to_name)
            if conversation_text:
                conversation_lines.append(conversation_text)
            
            # Collect user statistics
            # for user_id in getattr(memcell, "user_id_list", []) or []:
            for user_id in user_id_to_name.keys():
                if user_id not in user_profiles:
                    user_profiles[user_id] = {
                        "user_id": user_id,
                        "user_name": user_id_to_name.get(user_id, "Unknown"),
                        "message_count": 0
                    }
                user_profiles[user_id]["message_count"] += 1
        
        if not conversation_lines:
            logger.warning("[ProfileMemoryExtractor] No conversation text to analyze for companion profiles")
            return None
        
        conversation_text = "\n".join(conversation_lines)
        logger.info(f"[ProfileMemoryExtractor] Built companion conversation with {len(conversation_lines)} segments")
        logger.info(f"[ProfileMemoryExtractor] Found {len(user_profiles)} unique users for companion analysis")
        
        # Retrieve old profile information if available
        old_profiles_map: Dict[str, ProfileMemory] = {}
        if request.old_memory_list:
            for mem in request.old_memory_list:
                if mem.memory_type == MemoryType.PROFILE and hasattr(mem, 'user_id'):
                    old_profiles_map[mem.user_id] = mem
        
        # Extract Part3 profiles for each user
        companion_profiles: List[ProfileMemory] = []
        
        for user_id, user_info in user_profiles.items():
            logger.info(
                f"[ProfileMemoryExtractor] Analyzing companion profile for: {user_info['user_name']} "
                f"({user_info['message_count']} messages)"
            )
            
            # Build Part3 prompt
            prompt = CONVERSATION_PROFILE_PART3_EXTRACTION_PROMPT
            prompt += f"\n\n**Existing User Profile:**\n"
            prompt += f"User ID: {user_id}\n"
            prompt += f"User Name: {user_info['user_name']}\n"
            
            # Add old profile information if available
            if user_id in old_profiles_map:
                old_profile = old_profiles_map[user_id]
                prompt += f"\n**Previous Profile Summary:**\n"
                if hasattr(old_profile, 'personality') and old_profile.personality:
                    prompt += f"Personality: {old_profile.personality}\n"
                if hasattr(old_profile, 'soft_skills') and old_profile.soft_skills:
                    prompt += f"Soft Skills: {old_profile.soft_skills}\n"
            
            prompt += f"\n**New Conversation:**\n{conversation_text}\n"
            prompt += f"\n**Please extract the personality dimensions for user: {user_info['user_name']}**"
            
            # Call LLM for analysis
            try:
                response_text = await self.llm_provider.generate(prompt, temperature=0.3)
                logger.info(
                    f"[ProfileMemoryExtractor] Successfully extracted companion profile for {user_info['user_name']}"
                )
                
                # Create ProfileMemory object with 90-dimension analysis stored in personality field
                from datetime import datetime
                profile_memory = ProfileMemory(
                    memory_type=MemoryType.PROFILE,
                    user_id=user_id,
                    timestamp=datetime.now(),
                    ori_event_id_list=[memcell.event_id for memcell in request.memcell_list if hasattr(memcell, 'event_id')],
                    group_id=request.group_id or "",
                    personality=[{
                        "value": "90-dimension-analysis",
                        "evidences": [],
                        "analysis": response_text
                    }],  # Store 90-dimension analysis in personality field
                    working_habit_preference=None,
                    soft_skills=None,
                    hard_skills=None,
                    work_responsibility=None,
                    tendency=None,
                    way_of_decision_making=None,
                    projects_participated=None,
                    group_importance_evidence=None
                )
                
                companion_profiles.append(profile_memory)
                
            except Exception as exc:
                logger.error(
                    f"[ProfileMemoryExtractor] Failed to extract companion profile for "
                    f"{user_info['user_name']}: {exc}"
                )
                continue
        
        if not companion_profiles:
            logger.warning("[ProfileMemoryExtractor] No companion profiles were successfully extracted")
            return None
        
        logger.info(f"[ProfileMemoryExtractor] Successfully extracted {len(companion_profiles)} companion profiles")
        return companion_profiles
