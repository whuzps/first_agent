"""
工厂模式：配置模型
"""
import os
import sys
from typing import List

from langchain_community.embeddings import DashScopeEmbeddings
# from sentence_transformers import SentenceTransformer

# 添加父目录到路径，确保可以导入 config
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from core.config import EMBEDDING_MODEL, DASHSCOPE_API_KEY
except ImportError:
    try:
        from service.config import EMBEDDING_MODEL, DASHSCOPE_API_KEY
    except ImportError:
        raise ImportError("无法导入 config 模块，请确保在正确的环境中运行")

class Model:
    # 单例缓存
    _dense_model = None
    _llm_model = None  # 【新增】

    @staticmethod
    def get_dense_embedding_model():
        """获取稠密向量模型 (DashScope)"""
        if Model._dense_model is None:
            # Model._dense_model = SentenceTransformer("moka-ai/m3e-base")
            Model._dense_model = DashScopeEmbeddings(
                model=EMBEDDING_MODEL,
                dashscope_api_key=DASHSCOPE_API_KEY
            )
        return Model._dense_model