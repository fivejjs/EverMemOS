"""Normalization helpers for profile memory extraction."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Set

from core.observation.logger import get_logger

from ...types import Memory, MemoryType
from .types import ProfileMemory, ProjectInfo

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Evidence helpers
# ---------------------------------------------------------------------------

def ensure_str_list(value: Any) -> List[str]:
    """Convert arbitrary values into a deduplicated list of stripped strings."""
    if not value:
        return []
    if isinstance(value, list):
        result: List[str] = []
        for item in value:
            if item is None:
                continue
            text = item.strip() if isinstance(item, str) else str(item).strip()
            if text and text not in result:
                result.append(text)
        return result
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    text = str(value).strip()
    return [text] if text else []


def format_evidence_entry(
    value: Any,
    *,
    default_date: Optional[str],
    conversation_date_map: Optional[Dict[str, str]],
) -> Optional[str]:
    """Format evidence entries to include the appropriate date prefix."""
    if value is None:
        return None
    item_str = value.strip() if isinstance(value, str) else str(value).strip()
    if not item_str:
        return None
    if "|" in item_str:
        return item_str

    conversation_id = conversation_id_from_evidence(item_str)
    if conversation_id:
        normalized_key = conversation_id
    elif "conversation_id" in item_str:
        normalized_key = item_str.split("conversation_id:")[-1].strip("[] ") or item_str
    else:
        normalized_key = item_str

    evidence_date: Optional[str] = None
    if conversation_id and conversation_date_map:
        evidence_date = conversation_date_map.get(conversation_id)
    if not evidence_date:
        evidence_date = default_date
    if evidence_date:
        return f"{evidence_date}|{normalized_key}"
    return normalized_key


def conversation_id_from_evidence(evidence: Any) -> Optional[str]:
    """Extract the conversation identifier from a formatted evidence entry."""
    if not isinstance(evidence, str):
        return None
    entry = evidence.strip()
    if not entry:
        return None
    if "|" in entry:
        entry = entry.split("|")[-1].strip()
    if "conversation_id:" in entry:
        entry = entry.split("conversation_id:")[-1]
    return entry.strip("[] ") or None


# ---------------------------------------------------------------------------
# Value helpers
# ---------------------------------------------------------------------------

def merge_value_with_evidences_lists(
    existing: Optional[List[Dict[str, Any]]],
    incoming: Optional[List[Dict[str, Any]]],
) -> Optional[List[Dict[str, Any]]]:
    """Merge two value/evidence lists while deduplicating evidences."""
    if not existing and not incoming:
        return None

    merged_map: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []

    def add_from(source: Optional[List[Dict[str, Any]]]) -> None:
        if not source:
            return
        for item in source:
            if not isinstance(item, dict) or not item:
                continue
            value = item.get("value", "")
            evidences = item.get("evidences", [])
            level_value = item.get("level", "")
            if not value:
                continue
            value_key = value.strip() if isinstance(value, str) else str(value).strip()
            if not value_key:
                continue
            if value_key not in merged_map:
                order.append(value_key)
                merged_map[value_key] = {"evidences": [], "level": ""}
            if evidences:
                for ev in evidences:
                    if ev and ev not in merged_map[value_key]["evidences"]:
                        merged_map[value_key]["evidences"].append(ev)
            level_str = (
                level_value.strip()
                if isinstance(level_value, str)
                else str(level_value).strip() if level_value is not None else ""
            )
            if level_str:
                merged_map[value_key]["level"] = level_str

    add_from(existing)
    add_from(incoming)

    if not order:
        return None

    merged_list: List[Dict[str, Any]] = []
    for val in order:
        entry = {"value": val, "evidences": merged_map[val]["evidences"]}
        level = merged_map[val].get("level", "")
        if level:
            entry["level"] = level
        merged_list.append(entry)
    return merged_list


def extract_values_with_evidence(
    raw_value: Any,
    *,
    valid_conversation_ids: Optional[Set[str]] = None,
    default_date: Optional[str] = None,
    conversation_date_map: Optional[Dict[str, str]] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Extract value/evidence pairs from heterogeneous LLM responses."""
    result: List[Dict[str, Any]] = []
    seen_values: Dict[str, Dict[str, Any]] = {}

    def add_entry(key: Any, evidence_list: Any, level: Any = None) -> None:
        if key is None:
            return
        value_str = key.strip() if isinstance(key, str) else str(key).strip()
        if not value_str:
            return

        evidence_items = ensure_str_list(evidence_list)
        formatted_evidences: List[str] = []
        for item in evidence_items:
            formatted = format_evidence_entry(
                item,
                default_date=default_date,
                conversation_date_map=conversation_date_map,
            )
            if not formatted:
                continue
            if valid_conversation_ids is not None:
                conversation_id = conversation_id_from_evidence(formatted)
                if not conversation_id or conversation_id not in valid_conversation_ids:
                    logger.warning(
                        "[ProfileMemoryExtractor] LLM Generated Unknown Conversation ID %s",
                        conversation_id,
                    )
                    continue
            formatted_evidences.append(formatted)

        level_str = (
            level.strip()
            if isinstance(level, str)
            else str(level).strip() if level is not None and level != "" else ""
        )

        if value_str not in seen_values:
            if not formatted_evidences:
                logger.warning(
                    "[ProfileMemoryExtractor] LLM returned value %s without evidences",
                    value_str,
                )
            seen_values[value_str] = {"evidences": [], "level": ""}

        entry = seen_values[value_str]
        if level_str:
            entry["level"] = level_str
        for ev in formatted_evidences:
            if ev not in entry["evidences"]:
                entry["evidences"].append(ev)

    if isinstance(raw_value, dict):
        if "value" in raw_value:
            add_entry(raw_value.get("value"), raw_value.get("evidences"), raw_value.get("level"))
        else:
            for key, evidence_list in raw_value.items():
                if key == "evidences":
                    continue
                add_entry(key, evidence_list, raw_value.get("level"))
    elif isinstance(raw_value, list):
        for entry in raw_value:
            if isinstance(entry, dict):
                if "value" in entry:
                    add_entry(
                        entry.get("value"),
                        entry.get("evidences"),
                        entry.get("level"),
                    )
                else:
                    evidence_source = entry.get("evidences") if "evidences" in entry else None
                    processed = False
                    for key, evidence_list in entry.items():
                        if key == "evidences":
                            continue
                        add_entry(
                            key,
                            evidence_list if evidence_source is None else evidence_source,
                            entry.get("level"),
                        )
                        processed = True
                    if not processed:
                        add_entry(entry, None, entry.get("level"))
            elif isinstance(entry, str):
                add_entry(entry, None)
            elif entry is not None:
                add_entry(entry, None)
    elif raw_value is not None:
        add_entry(raw_value, None)

    for value, stored in seen_values.items():
        entry = {"value": value, "evidences": stored.get("evidences", [])}
        level_value = stored.get("level", "")
        if level_value:
            entry["level"] = level_value
        result.append(entry)

    return result or None


# ---------------------------------------------------------------------------
# Skill helpers
# ---------------------------------------------------------------------------

def process_skills(
    raw_value: Any,
    *,
    extract_evidences: bool = False,
    valid_conversation_ids: Optional[Set[str]] = None,
    default_date: Optional[str] = None,
    conversation_date_map: Optional[Dict[str, str]] = None,
) -> Optional[List[Dict[str, Any]]]:
    """
    Normalize skill data and attach evidences when requested.

    Returns a list shaped like::
        [{"skill": "Python", "level": "高级", "evidences": ["id1"]}]
    """
    if not raw_value:
        return None

    skills: List[Dict[str, Any]] = []
    skill_names = set()

    raw_items: List[Any]
    if isinstance(raw_value, list):
        raw_items = raw_value
    elif raw_value is None:
        raw_items = []
    else:
        raw_items = [raw_value]

    def add_skill(skill_name: Any, level: Any, evidence_list: Any = None) -> None:
        if skill_name is None:
            return
        name_str = skill_name.strip() if isinstance(skill_name, str) else str(skill_name).strip()
        if not name_str or name_str in skill_names:
            return
        level_str = (
            level.strip()
            if isinstance(level, str)
            else str(level).strip() if level is not None else ""
        )

        skill_dict = {"skill": name_str, "level": level_str}

        if extract_evidences:
            evidence_items = ensure_str_list(evidence_list)
            formatted_evidences: List[str] = []
            for evidence in evidence_items:
                formatted = format_evidence_entry(
                    evidence,
                    default_date=default_date,
                    conversation_date_map=conversation_date_map,
                )
                if not formatted:
                    continue
                if valid_conversation_ids is not None:
                    conversation_id = conversation_id_from_evidence(formatted)
                    if not conversation_id or conversation_id not in valid_conversation_ids:
                        logger.warning(
                            "[ProfileMemoryExtractor] LLM Generated Unknown Conversation ID %s",
                            conversation_id,
                        )
                        continue
                formatted_evidences.append(formatted)
            skill_dict["evidences"] = formatted_evidences
        else:
            skill_dict["evidences"] = []

        skills.append(skill_dict)
        skill_names.add(name_str)

    for item in raw_items:
        skill_name: Optional[str] = None
        level_value: Optional[str] = None
        evidence_list: Any = None

        if isinstance(item, dict):
            if "skill" in item:
                skill_name = item.get("skill")
                level_value = item.get("level", "")
                evidence_list = item.get("evidences") if extract_evidences else None
            else:
                evidence_list = item.get("evidences") if extract_evidences else None
                for key, value in item.items():
                    if key == "evidences":
                        continue
                    skill_name = key
                    level_value = value
                    break
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            skill_name, level_value = item[0], item[1]
            if extract_evidences and len(item) > 2:
                evidence_list = item[2]
        elif isinstance(item, str):
            skill_name = item
            level_value = ""
        elif item is not None:
            skill_name = item
            level_value = ""

        add_skill(skill_name, level_value, evidence_list)

    return skills or None


def extract_skill_list(raw_value: Any) -> Optional[List[Dict[str, Any]]]:
    """Extract skills without evidence processing (legacy support)."""
    return process_skills(raw_value, extract_evidences=False)


def normalize_skills_with_evidence(
    raw_value: Any,
    *,
    valid_conversation_ids: Optional[Set[str]] = None,
    default_date: Optional[str] = None,
    conversation_date_map: Optional[Dict[str, str]] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Process skills and include evidences when available."""
    return process_skills(
        raw_value,
        extract_evidences=True,
        valid_conversation_ids=valid_conversation_ids,
        default_date=default_date,
        conversation_date_map=conversation_date_map,
    )


def merge_skill_lists(
    existing: Optional[List[Dict[str, Any]]],
    incoming: Optional[List[Dict[str, Any]]],
) -> Optional[List[Dict[str, Any]]]:
    """Merge two skill lists with evidence aggregation."""
    if not existing and not incoming:
        return None

    merged_map: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []

    def add_from(source: Optional[List[Dict[str, Any]]]) -> None:
        if not source:
            return
        for item in source:
            if not isinstance(item, dict) or not item:
                continue
            skill_name = item.get("skill", "")
            level_value = item.get("level", "")
            evidences = item.get("evidences", [])
            if not skill_name:
                continue
            skill_key = skill_name.strip()
            level_str = (
                level_value.strip()
                if isinstance(level_value, str)
                else str(level_value).strip() if level_value is not None else ""
            )
            if skill_key not in merged_map:
                order.append(skill_key)
                merged_map[skill_key] = {"level": level_str, "evidences": []}
            else:
                if level_str:
                    merged_map[skill_key]["level"] = level_str

            if evidences:
                for ev in evidences:
                    if ev and ev not in merged_map[skill_key]["evidences"]:
                        merged_map[skill_key]["evidences"].append(ev)

    add_from(existing)
    add_from(incoming)

    if not order:
        return None

    return [
        {"skill": skill, "level": merged_map[skill]["level"], "evidences": merged_map[skill]["evidences"]}
        for skill in order
    ]


# ---------------------------------------------------------------------------
# Project helpers
# ---------------------------------------------------------------------------

def project_to_dict(project: ProjectInfo | Dict[str, Any]) -> Dict[str, Any]:
    """Serialize ProjectInfo for prompt payloads."""
    if isinstance(project, ProjectInfo):
        return {
            "project_id": project.project_id,
            "project_name": project.project_name,
            "entry_date": project.entry_date,
            "subtasks": project.subtasks or [],
            "user_objective": project.user_objective or [],
            "contributions": project.contributions or [],
            "user_concerns": project.user_concerns or [],
        }
    return {
        "project_id": project.get("project_id", ""),
        "project_name": project.get("project_name", ""),
        "entry_date": project.get("entry_date", ""),
        "subtasks": project.get("subtasks", []),
        "user_objective": project.get("user_objective", []),
        "contributions": project.get("contributions", []),
        "user_concerns": project.get("user_concerns", []),
    }


def convert_projects_to_dataclass(
    projects_data: Iterable[Any],
    *,
    valid_conversation_ids: Optional[Set[str]] = None,
    default_date: Optional[str] = None,
    conversation_date_map: Optional[Dict[str, str]] = None,
) -> List[ProjectInfo]:
    """Convert project payloads into ProjectInfo dataclasses."""
    projects: List[ProjectInfo] = []
    for project_data in projects_data:
        if isinstance(project_data, ProjectInfo):
            projects.append(
                ProjectInfo(
                    project_id=project_data.project_id,
                    project_name=project_data.project_name,
                    entry_date=project_data.entry_date,
                    subtasks=list(project_data.subtasks) if project_data.subtasks else None,
                    user_objective=list(project_data.user_objective) if project_data.user_objective else None,
                    contributions=list(project_data.contributions) if project_data.contributions else None,
                    user_concerns=list(project_data.user_concerns) if project_data.user_concerns else None,
                )
            )
            continue
        if not isinstance(project_data, dict):
            continue

        project_id = str(project_data.get("project_id") or "").strip()
        project_name = str(project_data.get("project_name") or "").strip()
        entry_date = _normalize_entry_date(project_data.get("entry_date"))

        subtasks = _normalize_project_field(
            project_data.get("subtasks"),
            valid_conversation_ids=valid_conversation_ids,
            default_date=default_date,
            conversation_date_map=conversation_date_map,
        )
        user_objective = _normalize_project_field(
            project_data.get("user_objective"),
            valid_conversation_ids=valid_conversation_ids,
            default_date=default_date,
            conversation_date_map=conversation_date_map,
        )
        contributions = _normalize_project_field(
            project_data.get("contributions"),
            valid_conversation_ids=valid_conversation_ids,
            default_date=default_date,
            conversation_date_map=conversation_date_map,
        )
        user_concerns = _normalize_project_field(
            project_data.get("user_concerns"),
            valid_conversation_ids=valid_conversation_ids,
            default_date=default_date,
            conversation_date_map=conversation_date_map,
        )

        projects.append(
            ProjectInfo(
                project_id=project_id,
                project_name=project_name,
                entry_date=entry_date,
                subtasks=subtasks or None,
                user_objective=user_objective or None,
                contributions=contributions or None,
                user_concerns=user_concerns or None,
            )
        )
    return projects


def _normalize_project_field(
    value: Any,
    *,
    valid_conversation_ids: Optional[Set[str]],
    default_date: Optional[str],
    conversation_date_map: Optional[Dict[str, str]],
) -> List[Dict[str, Any]]:
    if value and isinstance(value, list) and value:
        # Always process through extract_values_with_evidence to ensure
        # conversation_id validation and evidence formatting
        return extract_values_with_evidence(
            value,
            valid_conversation_ids=valid_conversation_ids,
            default_date=default_date,
            conversation_date_map=conversation_date_map,
        ) or []
    return []


def _normalize_entry_date(value: Any) -> str:
    """Return a YYYY-MM-DD date string or empty string when invalid."""
    if value is None:
        return ""
    entry_date = str(value).strip()
    if not entry_date:
        return ""
    try:
        datetime.strptime(entry_date, "%Y-%m-%d")
    except ValueError:
        logger.debug(
            "[ProfileMemoryExtractor] Invalid entry_date `%s`; resetting to empty",
            value,
        )
        return ""
    return entry_date


def merge_projects_participated(
    existing_projects: Optional[List[ProjectInfo]],
    incoming_projects: Optional[List[ProjectInfo]],
) -> List[ProjectInfo]:
    """Merge project participation lists, deduplicating by project id/name."""

    def clone_project(project: ProjectInfo) -> ProjectInfo:
        return ProjectInfo(
            project_id=project.project_id,
            project_name=project.project_name,
            entry_date=project.entry_date,
            subtasks=list(project.subtasks) if project.subtasks else None,
            user_objective=list(project.user_objective) if project.user_objective else None,
            contributions=list(project.contributions) if project.contributions else None,
            user_concerns=list(project.user_concerns) if project.user_concerns else None,
        )

    merged_projects: List[ProjectInfo] = [clone_project(project) for project in existing_projects or []]

    for project in incoming_projects or []:
        match: Optional[ProjectInfo] = None
        for existing_project in merged_projects:
            if project.project_id and existing_project.project_id:
                if project.project_id == existing_project.project_id:
                    match = existing_project
                    break
            elif project.project_name and existing_project.project_name:
                if project.project_name == existing_project.project_name:
                    match = existing_project
                    break

        if match:
            match.entry_date = match.entry_date or project.entry_date
            match.subtasks = merge_value_with_evidences_lists(match.subtasks, project.subtasks)
            match.user_objective = merge_value_with_evidences_lists(
                match.user_objective,
                project.user_objective,
            )
            match.contributions = merge_value_with_evidences_lists(
                match.contributions,
                project.contributions,
            )
            match.user_concerns = merge_value_with_evidences_lists(
                match.user_concerns,
                project.user_concerns,
            )
        else:
            merged_projects.append(clone_project(project))

    return merged_projects


# ---------------------------------------------------------------------------
# Profile helpers
# ---------------------------------------------------------------------------

def remove_evidences_from_profile(profile_obj: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively remove evidence fields to keep prompts concise."""
    def strip_content(content: Any) -> Any:
        if isinstance(content, dict):
            return {
                key: strip_content(value)
                for key, value in content.items()
                if key != "evidences"
            }
        if isinstance(content, list):
            return [strip_content(item) for item in content]
        return content

    result: Dict[str, Any] = {}
    for key, value in profile_obj.items():
        if key == "evidences":
            continue
        result[key] = strip_content(value)
    return result


def accumulate_old_memory_entry(
    memory: Memory,
    participants_profile_list: List[Dict[str, Any]],
) -> None:
    """Convert legacy Memory objects into prompt-ready dictionaries."""
    try:
        if memory.memory_type != MemoryType.PROFILE:
            return

        profile_obj: Dict[str, Any] = {"user_id": memory.user_id}

        if getattr(memory, "user_name", None):
            profile_obj["user_name"] = memory.user_name

        # 直接使用原始数据，保留 evidences（不使用 extract_skill_list，它会丢弃 evidences）
        hard_skills = getattr(memory, "hard_skills", None)
        if hard_skills:
            profile_obj["hard_skills"] = hard_skills

        soft_skills = getattr(memory, "soft_skills", None)
        if soft_skills:
            profile_obj["soft_skills"] = soft_skills

        for field_name in (
            "motivation_system",
            "fear_system",
            "value_system",
            "humor_use",
            "colloquialism",
        ):
            value = getattr(memory, field_name, None)
            if value:
                profile_obj[field_name] = value

        for field_name in (
            "way_of_decision_making",
            "personality",
            "user_goal",
            "work_responsibility",
            "working_habit_preference",
            "interests",
            "tendency",
        ):
            value = getattr(memory, field_name, None)
            if value:
                profile_obj[field_name] = value

        projects = getattr(memory, "projects_participated", None)
        if projects:
            project_payload = [
                project_to_dict(project)
                for project in projects
                if project is not None
            ]
            if project_payload:
                profile_obj["projects_participated"] = project_payload

        if len(profile_obj) > 1:
            participants_profile_list.append(profile_obj)

    except Exception as exc:  # pylint: disable=broad-except
        logger.error("[ProfileMemoryExtractor] Failed to extract old memory entry: %s", exc)


def profile_payload_to_memory(
    profile_data: Dict[str, Any],
    *,
    group_id: str,
    project_data: Optional[Dict[str, Any]] = None,
    valid_conversation_ids: Optional[Set[str]] = None,
    default_date: Optional[str] = None,
    conversation_date_map: Optional[Dict[str, str]] = None,
) -> Optional[ProfileMemory]:
    """Convert LLM payloads into ProfileMemory instances."""
    if not isinstance(profile_data, dict):
        return None

    extracted_user_id = str(profile_data.get("user_id", "")).strip()
    extracted_user_name = profile_data.get("user_name", "")
    if not extracted_user_id:
        logger.info(
            "[ProfileMemoryExtractor] LLM generated user %s has no user_id, skipping",
            extracted_user_name,
        )
        return None

    hard_skills = normalize_skills_with_evidence(
        profile_data.get("hard_skills"),
        valid_conversation_ids=valid_conversation_ids,
        default_date=default_date,
        conversation_date_map=conversation_date_map,
    )

    soft_skills = normalize_skills_with_evidence(
        profile_data.get("soft_skills"),
        valid_conversation_ids=valid_conversation_ids,
        default_date=default_date,
        conversation_date_map=conversation_date_map,
    )
    output_reasoning_raw = profile_data.get("output_reasoning")
    output_reasoning = None
    if output_reasoning_raw is not None:
        output_reasoning = str(output_reasoning_raw).strip() or None

    motivation_values = extract_values_with_evidence(
        profile_data.get("motivation_system"),
        valid_conversation_ids=valid_conversation_ids,
        default_date=default_date,
        conversation_date_map=conversation_date_map,
    )

    fear_values = extract_values_with_evidence(
        profile_data.get("fear_system"),
        valid_conversation_ids=valid_conversation_ids,
        default_date=default_date,
        conversation_date_map=conversation_date_map,
    )

    value_system_values = extract_values_with_evidence(
        profile_data.get("value_system"),
        valid_conversation_ids=valid_conversation_ids,
        default_date=default_date,
        conversation_date_map=conversation_date_map,
    )

    humor_values = extract_values_with_evidence(
        profile_data.get("humor_use"),
        valid_conversation_ids=valid_conversation_ids,
        default_date=default_date,
        conversation_date_map=conversation_date_map,
    )

    colloquialism_values = extract_values_with_evidence(
        profile_data.get("colloquialism"),
        valid_conversation_ids=valid_conversation_ids,
        default_date=default_date,
        conversation_date_map=conversation_date_map,
    )

    # role_responsibility 替代 user_goal，映射到 work_responsibility
    work_responsibility_values = extract_values_with_evidence(
        profile_data.get("role_responsibility"),
        valid_conversation_ids=valid_conversation_ids,
        default_date=default_date,
        conversation_date_map=conversation_date_map,
    )

    user_goal_values = extract_values_with_evidence(
        profile_data.get("user_goal"),
        valid_conversation_ids=valid_conversation_ids,
        default_date=default_date,
        conversation_date_map=conversation_date_map,
    )

    working_habit_values = extract_values_with_evidence(
        profile_data.get("working_habit_preference"),
        valid_conversation_ids=valid_conversation_ids,
        default_date=default_date,
        conversation_date_map=conversation_date_map,
    )

    interests_values = extract_values_with_evidence(
        profile_data.get("interests"),
        valid_conversation_ids=valid_conversation_ids,
        default_date=default_date,
        conversation_date_map=conversation_date_map,
    )

    tendency_source = profile_data.get("opinion_tendency")
    if tendency_source is None:
        tendency_source = profile_data.get("tendency")
    tendency_values = extract_values_with_evidence(
        tendency_source,
        valid_conversation_ids=valid_conversation_ids,
        default_date=default_date,
        conversation_date_map=conversation_date_map,
    )

    personality_values = extract_values_with_evidence(
        profile_data.get("personality"),
        valid_conversation_ids=valid_conversation_ids,
        default_date=default_date,
        conversation_date_map=conversation_date_map,
    )

    way_of_decision_values = extract_values_with_evidence(
        profile_data.get("way_of_decision_making"),
        valid_conversation_ids=valid_conversation_ids,
        default_date=default_date,
        conversation_date_map=conversation_date_map,
    )

    if project_data is not None:
        projects_participated = convert_projects_to_dataclass(
            project_data.get("projects_participated", []),
            valid_conversation_ids=valid_conversation_ids,
            default_date=default_date,
            conversation_date_map=conversation_date_map,
        )
    else:
        projects_participated = convert_projects_to_dataclass(
            profile_data.get("projects_participated", []),
            valid_conversation_ids=valid_conversation_ids,
            default_date=default_date,
            conversation_date_map=conversation_date_map,
        )

    if not (
        hard_skills
        or soft_skills
        or output_reasoning
        or motivation_values
        or fear_values
        or value_system_values
        or humor_values
        or colloquialism_values
        or way_of_decision_values
        or personality_values
        or projects_participated
        or user_goal_values
        or working_habit_values
        or interests_values
        or tendency_values
        or work_responsibility_values
    ):
        return None

    return ProfileMemory(
        memory_type=MemoryType.PROFILE,
        user_id=extracted_user_id,
        timestamp="",
        ori_event_id_list=[],
        group_id=group_id,
        user_name=extracted_user_name,
        hard_skills=hard_skills or None,
        soft_skills=soft_skills or None,
        output_reasoning=output_reasoning,
        motivation_system=motivation_values or None,
        fear_system=fear_values or None,
        value_system=value_system_values or None,
        humor_use=humor_values or None,
        colloquialism=colloquialism_values or None,
        way_of_decision_making=way_of_decision_values or None,
        personality=personality_values or None,
        projects_participated=projects_participated or None,
        user_goal=user_goal_values or None,
        work_responsibility=work_responsibility_values,
        working_habit_preference=working_habit_values or None,
        interests=interests_values or None,
        tendency=tendency_values or None,
        type=[MemoryType.PROFILE.value],
    )


def merge_single_profile(
    existing: ProfileMemory,
    new: ProfileMemory,
    *,
    group_id: str,
) -> ProfileMemory:
    """Merge two ProfileMemory objects with the same user id."""
    merged_hard_skills = merge_skill_lists(existing.hard_skills, new.hard_skills)
    merged_soft_skills = merge_skill_lists(existing.soft_skills, new.soft_skills)
    merged_motivation = merge_value_with_evidences_lists(
        existing.motivation_system,
        new.motivation_system,
    )
    merged_fear = merge_value_with_evidences_lists(existing.fear_system, new.fear_system)
    merged_value_system = merge_value_with_evidences_lists(
        existing.value_system,
        new.value_system,
    )
    merged_humor = merge_value_with_evidences_lists(existing.humor_use, new.humor_use)
    merged_colloquialism = merge_value_with_evidences_lists(
        existing.colloquialism,
        new.colloquialism,
    )
    merged_way_of_decision = merge_value_with_evidences_lists(
        existing.way_of_decision_making,
        new.way_of_decision_making,
    )
    merged_personality = merge_value_with_evidences_lists(existing.personality, new.personality)
    merged_user_goal = merge_value_with_evidences_lists(existing.user_goal, new.user_goal)
    merged_working_habits = merge_value_with_evidences_lists(
        existing.working_habit_preference,
        new.working_habit_preference,
    )
    merged_interests = merge_value_with_evidences_lists(existing.interests, new.interests)
    merged_tendency = merge_value_with_evidences_lists(existing.tendency, new.tendency)
    merged_work_responsibility = merge_value_with_evidences_lists(
        existing.work_responsibility,
        new.work_responsibility,
    )

    merged_projects = merge_projects_participated(existing.projects_participated, new.projects_participated)

    output_reasoning = new.output_reasoning if new.output_reasoning is not None else existing.output_reasoning

    merged_type: List[str] = []
    for source in (existing.type or [], new.type or []):
        for item in source:
            text = item.strip() if isinstance(item, str) else str(item).strip()
            if text and text not in merged_type:
                merged_type.append(text)
    if not merged_type:
        merged_type = [MemoryType.PROFILE.value]

    return ProfileMemory(
        memory_type=MemoryType.PROFILE,
        user_id=existing.user_id,
        timestamp=new.timestamp or existing.timestamp,
        ori_event_id_list=new.ori_event_id_list or existing.ori_event_id_list,
        user_name=new.user_name or existing.user_name,
        group_id=group_id or new.group_id or existing.group_id,
        hard_skills=merged_hard_skills,
        soft_skills=merged_soft_skills,
        output_reasoning=output_reasoning,
        motivation_system=merged_motivation,
        fear_system=merged_fear,
        value_system=merged_value_system,
        humor_use=merged_humor,
        colloquialism=merged_colloquialism,
        way_of_decision_making=merged_way_of_decision,
        personality=merged_personality,
        projects_participated=merged_projects or None,
        user_goal=merged_user_goal,
        work_responsibility=merged_work_responsibility,
        working_habit_preference=merged_working_habits,
        interests=merged_interests,
        tendency=merged_tendency,
        type=merged_type,
    )


def merge_profiles(
    profile_memories: Iterable[ProfileMemory],
    participants_profile_list: Iterable[Dict[str, Any]],
    *,
    group_id: str,
    default_date: Optional[str] = None,
    valid_conversation_ids: Optional[Set[str]] = None,
    conversation_date_map: Optional[Dict[str, str]] = None,
) -> List[ProfileMemory]:
    """Merge extracted profiles with existing participant profiles."""
    merged_dict: Dict[str, ProfileMemory] = {}

    for participant_profile in participants_profile_list:
        user_id = participant_profile.get("user_id")
        if not user_id:
            continue

        profile_memory = ProfileMemory(
            memory_type=MemoryType.PROFILE,
            user_id=user_id,
            timestamp="",
            ori_event_id_list=[],
            group_id=group_id,
            user_name=participant_profile.get("user_name"),
            hard_skills=normalize_skills_with_evidence(
                participant_profile.get("hard_skills"),
                valid_conversation_ids=None,
                default_date=None,
                conversation_date_map=None,
            )
            or None,
            soft_skills=normalize_skills_with_evidence(
                participant_profile.get("soft_skills"),
                valid_conversation_ids=None,
                default_date=None,
                conversation_date_map=None,
            )
            or None,
            motivation_system=participant_profile.get("motivation_system"),
            fear_system=participant_profile.get("fear_system"),
            value_system=participant_profile.get("value_system"),
            humor_use=participant_profile.get("humor_use"),
            colloquialism=participant_profile.get("colloquialism"),
            way_of_decision_making=participant_profile.get("way_of_decision_making"),
            personality=participant_profile.get("personality"),
            projects_participated=convert_projects_to_dataclass(
                participant_profile.get("projects_participated", []),
                valid_conversation_ids=None,
                default_date=None,
                conversation_date_map=None,
            )
            or None,
            user_goal=participant_profile.get("user_goal"),
            work_responsibility=participant_profile.get("work_responsibility"),
            working_habit_preference=participant_profile.get("working_habit_preference"),
            interests=participant_profile.get("interests"),
            tendency=participant_profile.get("tendency"),
            type=[MemoryType.PROFILE.value],
        )
        merged_dict[user_id] = profile_memory

    for new_profile in profile_memories:
        user_id = new_profile.user_id
        if user_id in merged_dict:
            existing_profile = merged_dict[user_id]
            merged_dict[user_id] = merge_single_profile(
                existing_profile,
                new_profile,
                group_id=group_id,
            )
        else:
            merged_dict[user_id] = new_profile

    return list(merged_dict.values())
