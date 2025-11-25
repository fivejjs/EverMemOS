"""
Hello World 任务

提供一个简单的 Hello World 任务
"""

from typing import Any

from core.asynctasks.task_manager import task


@task()
async def hello_world(data: Any) -> Any:
    return f"hello world: {data}"
