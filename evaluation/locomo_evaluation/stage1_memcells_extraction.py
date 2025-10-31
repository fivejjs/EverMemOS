from typing import Dict, List
import json
import os
import sys
import uuid
import asyncio
import time
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn,
    TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
)
from rich.console import Console
# Ensure project root is on sys.path so `src` can be imported when running directly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
SRC_DEV_DIR = os.path.join(PROJECT_ROOT, "src_dev")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if SRC_DEV_DIR not in sys.path:
    sys.path.insert(0, SRC_DEV_DIR)


from src.common_utils.datetime_utils import to_iso_format, from_iso_format, from_timestamp, get_now_with_timezone
from src.memory_layer.llm.llm_provider import LLMProvider
from src.memory_layer.memcell_extractor.base_memcell_extractor import RawData, MemCell
from src.memory_layer.memcell_extractor.conv_memcell_extractor import ConvMemCellExtractor, ConversationMemCellExtractRequest
from src.memory_layer.memory_extractor.episode_memory_extractor_locomo import EpisodeMemoryExtractRequest, EpisodeMemoryExtractor
from src.memory_layer.memory_extractor.event_log_extractor import EventLogExtractor
from src.memory_layer.types import RawDataType

from evaluation.locomo_evaluation.config import ExperimentConfig
from datetime import datetime, timedelta

def parse_locomo_timestamp(timestamp_str: str) -> datetime:
    """Parse LoComo timestamp format to datetime object."""
    timestamp_str = timestamp_str.replace("\\s+", " ").strip()
    dt = datetime.strptime(timestamp_str, "%I:%M %p on %d %B, %Y")
    return dt


def raw_data_load(locomo_data_path: str) -> Dict[str, List[RawData]]:
    with open(locomo_data_path, "r") as f:
        data = json.load(f)
    
    # data = [data[2]]
    # data = [data[0], data[1], data[2]]
    raw_data_dict = {}
    
    conversations = [data[i]['conversation'] for i in range(len(data))]
    print(f"   📅 Found {len(conversations)} conversations")
    for con_id, conversation in enumerate(conversations):
        messages = []
        # print(conversation.keys())
        session_keys = sorted(
            [
                key
                for key in conversation
                if key.startswith("session_") and not key.endswith("_date_time")
            ]
        )

        print(f"   📅 Found {len(session_keys)} sessions")
        print(
            f"   🎭 Speakers: {conversation.get('speaker_a', 'Unknown')} & {conversation.get('speaker_b', 'Unknown')}"
        )
        speaker_name_to_id = {}
        for session_key in session_keys:
            session_messages = conversation[session_key]
            session_time_key = f"{session_key}_date_time"

            if session_time_key in conversation:
                # Parse session timestamp
                session_time = parse_locomo_timestamp(conversation[session_time_key])

                # Process each message in this session
                for i, msg in enumerate(session_messages):
                    # Generate timestamp for this message (session time + message offset)
                    msg_timestamp = session_time + timedelta(
                        seconds=i * 30
                    )  # 30 seconds between messages
                    iso_timestamp = to_iso_format(msg_timestamp)

                    # Generate unique speaker_id for this conversation
                    speaker_name = msg["speaker"]
                    if speaker_name not in speaker_name_to_id:
                        # Generate unique ID: {name}_{conversation_index}
                        unique_id = f"{speaker_name.lower().replace(' ', '_')}_{con_id}"
                        speaker_name_to_id[speaker_name] = unique_id

                    # Process content with image information if present
                    content = msg["text"]
                    if msg.get("img_url"):
                        blip_caption = msg.get("blip_caption", "an image")
                        content = f"[{speaker_name} shared an image: {blip_caption}] {content}"

                    message = {
                        "speaker_id": speaker_name_to_id[speaker_name],
                        "user_name": speaker_name,
                        "speaker_name": speaker_name,
                        "content": content,
                        "timestamp": iso_timestamp,
                        "original_timestamp": conversation[session_time_key],
                        "dia_id": msg["dia_id"],
                        "session": session_key,
                    }
                    # Add optional fields if present
                    for optional_field in ["img_url", "blip_caption", "query"]:
                        if optional_field in msg:
                            message[optional_field] = msg[optional_field]
                    messages.append(message)
        raw_data_dict[str(con_id)] = messages

        print(f"   ✅ Converted {len(messages)} messages from {len(session_keys)} sessions")

    return raw_data_dict

def convert_conversation_to_raw_data_list(conversation: list) -> List[RawData]:
    raw_data_list = []
    for msg in conversation:
        raw_data_list.append(
            RawData(content=msg, data_id=str(uuid.uuid4()))
        )
    return raw_data_list

async def memcell_extraction_from_conversation(
    raw_data_list: List[RawData], 
    llm_provider: LLMProvider = None,
    memcell_extractor: ConvMemCellExtractor = None,
    smart_mask: bool = True,
    conv_id: str = None,  # 添加会话ID用于进度条描述
    progress: Progress = None,  # 添加进度条对象
    task_id: int = None  # 添加任务ID
) -> list:
    
    episode_extractor = EpisodeMemoryExtractor(llm_provider=llm_provider)
    memcell_list = []
    speakers = {
        raw_data.content["speaker_id"]
        for raw_data in raw_data_list
        if isinstance(raw_data.content, dict) and "speaker_id" in raw_data.content
    }
    history_raw_data_list = []
    # raw_data_list = raw_data_list[:100]
    
    # 处理消息
    total_messages = len(raw_data_list)
    smart_mask_flag = False

    for idx, raw_data in enumerate(raw_data_list):
        # 更新进度条（在处理前更新，显示正在处理第几条）
        if progress and task_id is not None:
            progress.update(task_id, completed=idx)
        
        if history_raw_data_list == [] or len(history_raw_data_list) == 1:
            history_raw_data_list.append(raw_data)
            continue

        if smart_mask and len(history_raw_data_list) > 5:
            smart_mask_flag = True
            # analysis_history = history_raw_data_list[:-1]
        else:
            # analysis_history = history_raw_data_list
            smart_mask_flag = False
        request = ConversationMemCellExtractRequest(
            history_raw_data_list=history_raw_data_list,
            new_raw_data_list=[raw_data],
            user_id_list=list(speakers),
            smart_mask_flag=smart_mask_flag,
            # group_id="group_1",
        )
        for i in range(5):
            try:
                result = await memcell_extractor.extract_memcell(request)
                break
            except Exception as e:
                print('retry: ', i)
                if i == 4:
                    raise Exception("Memcell extraction failed")
                continue
        memcell_result = result[0]
        # print(f"   ✅ Memcell result: {memcell_result}")  # 注释掉避免干扰进度条
        if memcell_result is None:
            history_raw_data_list.append(raw_data)
        elif isinstance(memcell_result, MemCell):
            if smart_mask_flag:
                history_raw_data_list = [history_raw_data_list[-1], raw_data]
            else:
                history_raw_data_list = [raw_data]
            memcell_result.summary = memcell_result.episode[:200] + "..."
            memcell_list.append(memcell_result)
        else:
            console = Console()
            console.print("--------------------------------")
            console.print(f"   ❌ Memcell result: {memcell_result}", style="bold red")
            raise Exception("Memcell extraction failed")
    
    # 处理完成，更新进度为100%
    if progress and task_id is not None:
        progress.update(task_id, completed=total_messages)
    
    if history_raw_data_list:
        memcell = (MemCell(
            type=RawDataType.CONVERSATION,
            event_id=str(uuid.uuid4()),
            user_id_list=list(speakers),
            original_data=history_raw_data_list,
            timestamp=(memcell_list[-1].timestamp),
            summary="111",
        ))
        episode_request = EpisodeMemoryExtractRequest(
            memcell_list=[memcell],
            user_id_list=request.user_id_list,
            participants=list(speakers),
            group_id=request.group_id
        )
                
        episode_result = await episode_extractor.extract_memory(episode_request, use_group_prompt=True)
        memcell.episode = episode_result.episode
        memcell.subject = episode_result.subject
        memcell.summary = episode_result.episode[:200] + "..."
        memcell.original_data = episode_extractor.get_conversation_text(history_raw_data_list)
        original_data_list = []
        for raw_data in history_raw_data_list:
            original_data_list.append(memcell_extractor._data_process(raw_data))
        memcell.original_data = original_data_list
        memcell_list.append(memcell)

    return memcell_list

async def process_single_conversation(
    conv_id: str, 
    conversation: list, 
    save_dir: str,
    llm_provider: LLMProvider = None,
    event_log_extractor: EventLogExtractor = None,  # 添加event_log_extractor参数
    progress_counter: dict = None,  # 添加进度计数器
    progress: Progress = None,  # 添加进度条对象
    task_id: int = None  # 添加任务ID
) -> tuple:
    """处理单个会话并返回结果
    
    Args:
        conv_id: 会话ID
        conversation: 会话数据
        save_dir: 保存目录
        llm_provider: 共享的LLM提供者实例
        event_log_extractor: 事件日志提取器实例
        progress: 进度条对象
        task_id: 进度任务ID
        
    Returns:
        tuple: (conv_id, memcell_list)
    """
    try:
        # 更新状态为处理中
        if progress and task_id is not None:
            progress.update(task_id, status="处理中")
        
        raw_data_list = convert_conversation_to_raw_data_list(conversation)
        memcell_extractor = ConvMemCellExtractor(llm_provider=llm_provider)
        memcell_list = await memcell_extraction_from_conversation(
            raw_data_list, 
            llm_provider=llm_provider,
            memcell_extractor=memcell_extractor,
            conv_id=conv_id,  # 传递会话ID
            progress=progress,  # 传递进度条对象
            task_id=task_id  # 传递任务ID
        )
        # print(f"   ✅ 会话 {conv_id}: {len(memcell_list)} memcells extracted")  # 注释掉避免干扰进度条
        
        # 在保存前转换时间戳为 datetime 对象
        for memcell in memcell_list:
            if hasattr(memcell, 'timestamp'):
                ts = memcell.timestamp
                if isinstance(ts, (int, float)):
                    # 将 int/float 时间戳转换为带时区的 datetime
                    memcell.timestamp = from_timestamp(ts)
                elif isinstance(ts, str):
                    # 将字符串时间戳转换为带时区的 datetime
                    memcell.timestamp = from_iso_format(ts)
                elif not isinstance(ts, datetime):
                    # 如果不是预期的类型，使用当前时间
                    memcell.timestamp = get_now_with_timezone()
        
        # 生成event log（为每个memcell提取event log）
        if event_log_extractor:
            for memcell in memcell_list:
                if hasattr(memcell, 'episode') and memcell.episode:
                    try:
                        event_log = await event_log_extractor.extract_event_log(
                            episode_text=memcell.episode,
                            timestamp=memcell.timestamp
                        )
                        if event_log:
                            # 将event_log添加到memcell的字典中
                            memcell.event_log = event_log
                    except Exception as e:
                        console = Console()
                        console.print(f"\n⚠️  生成event log失败 (Conv {conv_id}): {e}", style="yellow")
                        # 继续处理，即使event log生成失败
        
        # 保存单个会话的结果
        memcell_dicts = []
        for memcell in memcell_list:
            memcell_dict = memcell.to_dict()
            # 如果有event_log，添加到字典中
            if hasattr(memcell, 'event_log') and memcell.event_log:
                memcell_dict['event_log'] = memcell.event_log.to_dict()
            memcell_dicts.append(memcell_dict)

        memcell_dicts = [memcell_dict for memcell_dict in memcell_dicts ]
        print(memcell_dicts)
        output_file = os.path.join(save_dir, f"memcell_list_conv_{conv_id}.json")
        with open(output_file, "w") as f:
            json.dump(memcell_dicts, f, ensure_ascii=False, indent=2)

        # 更新进度（静默，避免干扰进度条）
        if progress_counter:
            progress_counter['completed'] += 1
            # 不打印，避免干扰进度条
        
        return conv_id, memcell_list
        
    except Exception as e:
        # 显示错误信息，这样我们能知道具体问题
        console = Console()
        console.print(f"\n❌ 处理会话 {conv_id} 时出错: {e}", style="bold red")
        if progress_counter:
            progress_counter['completed'] += 1
            progress_counter['failed'] += 1
        import traceback
        traceback.print_exc()
        return conv_id, []


async def main():
    """主函数 - 并发处理所有会话"""
    
    config = ExperimentConfig()
    llm_service = config.llm_service
    dataset_path = config.datase_path
    raw_data_dict = raw_data_load(dataset_path)
    
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(CURRENT_DIR, "results"), exist_ok=True)
    os.makedirs(os.path.join(CURRENT_DIR, "results", config.experiment_name), exist_ok=True)
    os.makedirs(os.path.join(CURRENT_DIR, "results", config.experiment_name, "memcells"), exist_ok=True)
    save_dir = os.path.join(CURRENT_DIR, "results", config.experiment_name, "memcells")
    
    console = Console()
    console.print(f"\n📊 总共发现 {len(raw_data_dict)} 个会话", style="bold cyan")
    total_messages = sum(len(conv) for conv in raw_data_dict.values())
    console.print(f"📝 总消息数: {total_messages}", style="bold blue")
    console.print(f"🚀 开始并发处理所有会话...\n", style="bold green")
    
    # 创建共享的 LLM Provider 和 MemCell Extractor 实例（解决连接竞争问题）
    console.print("⚙️ 初始化 LLM Provider...", style="yellow")
    console.print(f"   模型: {config.llm_config[llm_service]['model']}", style="dim")
    console.print(f"   Base URL: {config.llm_config[llm_service]['base_url']}", style="dim")
    
    shared_llm_provider = LLMProvider(
        provider_type="openai",
        model=config.llm_config[llm_service]["model"],
        api_key=config.llm_config[llm_service]["api_key"],
        base_url=config.llm_config[llm_service]["base_url"],
        temperature=config.llm_config[llm_service]["temperature"],
        max_tokens=config.llm_config[llm_service]["max_tokens"],
    )
    
    # 创建共享的 Event Log Extractor
    console.print("⚙️ 初始化 Event Log Extractor...", style="yellow")
    shared_event_log_extractor = EventLogExtractor(llm_provider=shared_llm_provider)
    
    # 创建进度计数器
    progress_counter = {
        'total': len(raw_data_dict),
        'completed': 0,
        'failed': 0
    }
    
    # 使用 Rich 进度条
    start_time = time.time()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),  # 显示 "3/10" 格式
        TextColumn("•"),
        TaskProgressColumn(),  # 显示百分比
        TextColumn("•"),
        TimeElapsedColumn(),  # 已用时间
        TextColumn("•"),
        TimeRemainingColumn(),  # 预计剩余时间
        TextColumn("•"),
        TextColumn("[bold blue]{task.fields[status]}"),
        console=console,
        transient=False,
        refresh_per_second=1
    ) as progress:
        # 创建主进度任务
        main_task = progress.add_task(
            "[bold cyan]🎯 总进度", 
            total=len(raw_data_dict),
            completed=0,  # 初始化为0
            status="处理中"
        )
        
        # 为每个会话创建独立的进度条任务
        conversation_tasks = {}
        updated_tasks = []
        
        for conv_id, conversation in raw_data_dict.items():
            # 创建每个会话的进度条
            conv_task_id = progress.add_task(
                f"[yellow]Conv-{conv_id}",  # 简化名称
                total=len(conversation),  # 消息总数
                completed=0,  # 初始化为0
                status="等待"
            )
            conversation_tasks[conv_id] = conv_task_id
            
            # 创建处理任务
            task = process_single_conversation(
                conv_id, 
                conversation, 
                save_dir,
                llm_provider=shared_llm_provider,
                event_log_extractor=shared_event_log_extractor,  # 传递event_log_extractor
                progress_counter=progress_counter,
                progress=progress,
                task_id=conv_task_id
            )
            updated_tasks.append(task)
            
        
        # 定义完成时更新函数
        async def run_with_completion(task, conv_id):
            result = await task
            progress.update(conversation_tasks[conv_id], 
                          status="✅",
                          completed=progress.tasks[conversation_tasks[conv_id]].total)
            progress.update(main_task, advance=1)
            return result
        
        # 并发执行所有任务
        results = await asyncio.gather(*[
            run_with_completion(task, conv_id) 
            for (conv_id, _), task in zip(raw_data_dict.items(), updated_tasks)
        ])
        # with open(os.path.join(save_dir, "response_info.json"), "w") as f:
        #     json.dump(shared_llm_provider.provider.response_info, f, ensure_ascii=False, indent=2)
        # 更新主进度为完成
        progress.update(main_task, status="✅ 完成")
    
    end_time = time.time()
    
    # 统计结果
    all_memcells = []
    successful_convs = 0
    for conv_id, memcell_list in results:
        if memcell_list:
            successful_convs += 1
            all_memcells.extend(memcell_list)
    
    console.print("\n" + "="*60, style="dim")
    console.print("📊 处理完成统计:", style="bold")
    console.print(f"   ✅ 成功处理会话数: {successful_convs}/{len(raw_data_dict)}", style="green")
    console.print(f"   📝 总共提取的 memcells: {len(all_memcells)}", style="blue")
    console.print(f"   ⏱️  总耗时: {end_time - start_time:.2f} 秒", style="yellow")
    console.print(f"   🚀 平均每会话耗时: {(end_time - start_time)/len(raw_data_dict):.2f} 秒", style="cyan")
    console.print("="*60, style="dim")
    
    # 保存汇总结果
    all_memcells_dicts = [memcell.to_dict() for memcell in all_memcells]
    summary_file = os.path.join(save_dir, "memcell_list_all.json")
    with open(summary_file, "w") as f:
        json.dump(all_memcells_dicts, f, ensure_ascii=False, indent=2)
    console.print(f"\n💾 汇总结果已保存到: {summary_file}", style="green")
    
    # 保存处理摘要
    summary = {
        "total_conversations": len(raw_data_dict),
        "successful_conversations": successful_convs,
        "total_memcells": len(all_memcells),
        "processing_time_seconds": end_time - start_time,
        "average_time_per_conversation": (end_time - start_time)/len(raw_data_dict),
        "conversation_results": {
            conv_id: len(memcell_list) 
            for conv_id, memcell_list in results
        }
    }
    summary_info_file = os.path.join(save_dir, "processing_summary.json")
    with open(summary_info_file, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    console.print(f"📊 处理摘要已保存到: {summary_info_file}\n", style="green")


if __name__ == "__main__":
    asyncio.run(main())