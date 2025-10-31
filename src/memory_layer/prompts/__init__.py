
"""
多语言提示词模块

通过环境变量 MEMORY_LANGUAGE 控制语言，支持 'en' 和 'zh'
默认使用英文 ('en')

使用方法：
1. 设置环境变量：export MEMORY_LANGUAGE=zh
2. 现有代码无需修改，直接导入使用
"""

import os

# 获取语言设置，默认为英文
MEMORY_LANGUAGE = os.getenv('MEMORY_LANGUAGE', 'en').lower()

# 支持的语言
SUPPORTED_LANGUAGES = ['en', 'zh']

if MEMORY_LANGUAGE not in SUPPORTED_LANGUAGES:
    print(f"Warning: Unsupported language '{MEMORY_LANGUAGE}', falling back to 'en'")
    MEMORY_LANGUAGE = 'en'

# 根据语言设置导入提示词
if MEMORY_LANGUAGE == 'zh':
    # 中文提示词
    from .zh.conv_prompts import CONV_BOUNDARY_DETECTION_PROMPT, CONV_SUMMARY_PROMPT
    from .zh.episode_mem_prompts import EPISODE_GENERATION_PROMPT
    from .zh.profile_mem_prompts import CONVERSATION_PROFILE_EXTRACTION_PROMPT
    from .zh.email_prompts import EMAIL_BOUNDARY_DETECTION_PROMPT, EMAIL_SUMMARY_PROMPT
    from .zh.linkdoc_prompts import LINKDOC_BOUNDARY_DETECTION_PROMPT, LINKDOC_SUMMARY_PROMPT
else:
    # 英文提示词（默认）
    from .en.conv_prompts import CONV_BOUNDARY_DETECTION_PROMPT, CONV_SUMMARY_PROMPT
    from .en.episode_mem_prompts import EPISODE_GENERATION_PROMPT, GROUP_EPISODE_GENERATION_PROMPT
    from .en.profile_mem_prompts import CONVERSATION_PROFILE_EXTRACTION_PROMPT

# 导出当前语言信息
CURRENT_LANGUAGE = MEMORY_LANGUAGE

def get_current_language():
    """获取当前语言"""
    return CURRENT_LANGUAGE

def set_language(language: str):
    """设置语言（需要重启应用才能生效）"""
    global MEMORY_LANGUAGE, CURRENT_LANGUAGE
    if language.lower() in SUPPORTED_LANGUAGES:
        MEMORY_LANGUAGE = language.lower()
        CURRENT_LANGUAGE = MEMORY_LANGUAGE
        print(f"Language set to: {MEMORY_LANGUAGE}")
    else:
        print(f"Unsupported language: {language}. Supported: {SUPPORTED_LANGUAGES}")
