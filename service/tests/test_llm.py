


import asyncio
import os
import traceback
from langchain_openai import ChatOpenAI

from core.config import DASHSCOPE_API_URL

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(model="qwen-flash", api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_API_URL)

import logging
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    from core.logging_config import setup_logging
    setup_logging()
    llm = get_llm()
    async def main():
        try:
            logger.info(await llm.ainvoke("你好"))
        except Exception as e:
            logger.error(traceback.format_exc())
    asyncio.run(main())