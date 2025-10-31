"""Utilities for merging profile memories collected from multiple groups."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from core.observation.logger import get_logger

from ...llm.llm_provider import LLMProvider
from .data_normalize import merge_single_profile, project_to_dict
from .types import ProfileMemory, ImportanceEvidence

logger = get_logger(__name__)


def convert_important_info_to_evidence(important_info: Dict[str, Any]) -> List[ImportanceEvidence]:
    """Convert aggregated group stats into ImportanceEvidence instances."""
    evidence_list: List[ImportanceEvidence] = []
    total_msgs = important_info["group_data"]["total_messages"]
    for user_id, user_data in important_info["user_data"].items():
        evidence_list.append(
            ImportanceEvidence(
                user_id=user_id,
                group_id=important_info["group_id"],
                speak_count=user_data["chat_count"],
                refer_count=user_data["at_count"],
                conversation_count=total_msgs,
            )
        )
    return evidence_list


class ProfileMemoryMerger:
    """Merge multiple ProfileMemory instances for a single user via LLM guidance."""

    def __init__(self, llm_provider: LLMProvider) -> None:
        if llm_provider is None:
            error_msg = "llm_provider must not be None"
            logger.exception(error_msg)
            raise ValueError(error_msg)
        self.llm_provider = llm_provider

    @staticmethod
    def _truncate_evidences(evidences: Iterable[Any]) -> List[str]:
        if not evidences:
            return []
        normalized = [str(item).strip() for item in evidences if item]
        normalized = [item for item in normalized if item]
        if len(normalized) <= 10:
            return normalized

        def parse_date(prefix: str) -> Optional[datetime]:
            if not prefix:
                return None
            try:
                return datetime.fromisoformat(prefix)
            except ValueError:
                try:
                    return datetime.strptime(prefix, "%Y-%m-%d")
                except ValueError:
                    return None

        records: List[Dict[str, Any]] = []
        for idx, entry in enumerate(normalized):
            date_value: Optional[datetime] = None
            has_date = False
            if "|" in entry:
                prefix = entry.split("|", 1)[0].strip()
                parsed = parse_date(prefix)
                if parsed is not None:
                    has_date = True
                    date_value = parsed
            records.append(
                {
                    "entry": entry,
                    "index": idx,
                    "has_date": has_date,
                    "date": date_value,
                }
            )

        # while len(records) > 10:
        #     removed = False
        #     for i, record in enumerate(records):
        #         if not record["has_date"]:
        #             del records[i]
        #             removed = True
        #             break
        #     if removed:
        #         continue

        #     oldest_index = 0
        #     oldest_date = records[0]["date"]
        #     for i in range(1, len(records)):
        #         current_date = records[i]["date"]
        #         if current_date is None:
        #             continue
        #         if oldest_date is None or current_date < oldest_date:
        #             oldest_date = current_date
        #             oldest_index = i
        #     del records[oldest_index]

        return [record["entry"] for record in records]

    @classmethod
    def _profile_memory_to_prompt_dict(cls, profile: ProfileMemory) -> Dict[str, Any]:
        """Convert ProfileMemory to dict format expected by the merge prompt."""

        def truncate_evidences_in_items(items: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
            if not items:
                return []
            result = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                item_copy = item.copy()
                evidences = item_copy.get("evidences", [])
                if evidences:
                    item_copy["evidences"] = cls._truncate_evidences(evidences)
                result.append(item_copy)
            return result

        return {
            "group_id": profile.group_id or "",
            "user_id": profile.user_id,
            "user_name": profile.user_name or "",
            "user_goal": truncate_evidences_in_items(profile.user_goal),
            "working_habit_preference": truncate_evidences_in_items(profile.working_habit_preference),
            "interests": truncate_evidences_in_items(profile.interests),
            "hard_skills": truncate_evidences_in_items(profile.hard_skills),
            "soft_skills": truncate_evidences_in_items(profile.soft_skills),
            "personality": truncate_evidences_in_items(profile.personality),
            "way_of_decision_making": truncate_evidences_in_items(profile.way_of_decision_making),
            "work_responsibility": truncate_evidences_in_items(profile.work_responsibility),
            "tendency": truncate_evidences_in_items(profile.tendency),
            "projects_participated": [
                project_to_dict(project)
                for project in profile.projects_participated or []
            ],
        }

    async def merge_group_profiles(
        self,
        group_profiles: List[ProfileMemory],
        user_id: str,
    ) -> ProfileMemory:
        if not group_profiles:
            error_msg = "group_profiles must not be empty when merging"
            logger.exception(error_msg)
            raise ValueError(error_msg)

        matching_profiles = [
            profile
            for profile in group_profiles
            if profile is not None and profile.user_id == user_id
        ]

        if not matching_profiles:
            error_msg = f"No ProfileMemory found for user_id '{user_id}' when merging"
            logger.exception(error_msg)
            raise ValueError(error_msg)

        base_profile = matching_profiles[0]
        merged_profile = merge_single_profile(
            base_profile,
            base_profile,
            group_id=base_profile.group_id or "",
        )

        for profile in matching_profiles[1:]:
            merge_group_id = profile.group_id or merged_profile.group_id or ""
            merged_profile = merge_single_profile(
                merged_profile,
                profile,
                group_id=merge_group_id,
            )

        return merged_profile
