import re
from typing import Optional
import dashscope as _dashscope

from core.config import DASHSCOPE_API_KEY

def transcribe_audio(audio: str, language: Optional[str] = "zh", enable_itn: bool = True) -> str:
    try:
        if not (_dashscope and audio):
            return ""
        from os import getenv
        _dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"
        messages = [{"role": "user", "content": [{"audio": audio}]}]
        response = _dashscope.MultiModalConversation.call(
            api_key=DASHSCOPE_API_KEY,
            model="qwen3-asr-flash",
            messages=messages,
            result_format="message",
            asr_options={"language": language or "zh", "enable_itn": bool(enable_itn)},
        )
        output = response.get("output") if isinstance(response, dict) else None
        if not output:
            return ""
        choices = output.get("choices") or []
        if not choices:
            return ""
        content = choices[0].get("message", {}).get("content") or []
        if not content:
            return ""
        item = content[0]
        text = item.get("text") if isinstance(item, dict) else None
        return str(text or "").strip()
    except Exception:
        return ""

def clean_input(text: str) -> str:
    """清理输入文本：去除多余空白与不可见字符。"""
    s = (text or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("\u200b", "")
    return s