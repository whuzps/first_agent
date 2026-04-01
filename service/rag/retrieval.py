import logging
import os
import sys
import time
import traceback
from typing import List, Dict, Any, Union

# 模块级 logger
logger = logging.getLogger(__name__)

from langchain_core.documents import Document

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 导入 Model
try:
    from model import Model
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "model", 
        os.path.join(current_dir, "model.py")
    )
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    Model = model_module.Model

# 导入 MilvusStore
try:
    from milvus_store import MilvusStore
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "milvus_store", 
        os.path.join(current_dir, "milvus_store.py")
    )
    milvus_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(milvus_module)
    MilvusStore = milvus_module.MilvusStore

# 导入 config
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

try:
    from core.config import DASHSCOPE_API_KEY, RERANK_MODEL
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "config", 
        os.path.join(parent_dir, "config.py")
    )
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    DASHSCOPE_API_KEY = config_module.DASHSCOPE_API_KEY
    RERANK_MODEL = config_module.RERANK_MODEL

from dashscope import TextReRank

# Rerank 分数阈值：低于此值的文档视为不相关，直接丢弃，防止低质量上下文引发幻觉
_RERANK_SCORE_THRESHOLD = float(os.getenv("RAG_RERANK_SCORE_THRESHOLD", "0.1"))

class Retrieval:
    def __init__(self):
        '''
        初始化稠密模型
        '''
        self.dense_model = Model.get_dense_embedding_model()

    def retrieve(self, query: str, collection_name: str, top_k: int = 3, debug: bool = False) -> Union[List[Document], Dict[str, Any]]:
        """
        检索
        """
        store = MilvusStore(collection_name=collection_name)
        
        # --- A. 向量化 ---
        # 计算稠密向量
        _start = time.perf_counter()
        query_dense = self.dense_model.embed_query(query)
        _end = time.perf_counter()
        logger.info(f"向量化耗时: {_end - _start:.4f}s")
        
        # 准备 debug 数据容器
        debug_info = {}

        # --- [测试使用] 独立执行单路检索 ---
        if debug:
            # 稠密单路：传向量
            debug_info['dense_only'] = self._search_single_channel(
                store, 
                data=[query_dense], 
                field="dense_vector", 
                metric="COSINE"
            )
            
            # 稀疏单路：[关键变化] 传原始文本
            # Milvus 会利用 Schema 里的 Function 自动将文本转为 BM25 稀疏向量进行搜索
            debug_info['sparse_only'] = self._search_single_channel(
                store, 
                data=[query], 
                field="sparse_vector", 
                metric="BM25",
                search_params={"drop_ratio_search": 0.2} # 仅查询高权重词，加速
            )
        _start = time.perf_counter()
        # --- B. 混合检索 (Recall) ---
        recall_k = top_k * 10 
        
        # 调用 MilvusStore 的混合检索 (注意参数变化)
        results = store.hybrid_search(
            query_text=query,             # 传文本 (用于稀疏)
            query_dense_vector=query_dense, # 传向量 (用于稠密)
            k=recall_k
        )
        _end = time.perf_counter()
        logger.info(f"混合检索耗时: {_end - _start:.4f}s")
        
        # 结果格式化 
        # (MilvusClient 返回的是 list of dicts，不是 Hit 对象)
        candidates = []
        if results:
            for hit in results[0]:
                # --- [修正] 解析逻辑 ---
                entity = hit.get('entity', {})
                
                # 1. 获取内容
                content = entity.get('text', "")
                
                # 2. 获取 Metadata (它是 entity 下的一个嵌套字典)
                # 优先从 entity['metadata'] 取，如果为空，尝试直接从 entity 取 (兼容旧数据)
                meta = entity.get('metadata', {})
                if not meta and 'source' in entity: 
                     meta = entity # 降级处理
                
                # 3. 获取分数 
                # 混合检索 RRF 后通常叫 'score'，单路检索叫 'distance'
                score = hit.get('score') 
                if score is None:
                    score = hit.get('distance')
                score = score or 0.0 # 兜底防止 None
                
                # 4. 补充 source (如果有单独字段)
                source = meta.get('source') or entity.get('source', "")
                
                # --- 组装 Document ---
                doc = Document(page_content=content, metadata=meta)
                doc.metadata["source"] = source
                doc.metadata["milvus_score"] = score
                # 保留原始分块文本，供后续构造 sources 展示使用
                doc.metadata["original_text"] = content
                
                candidates.append(doc)

        if debug:
            debug_info['rrf_candidates'] = candidates[:5]

        _start = time.perf_counter()
        # --- C. Rerank ---
        reranked_docs = self._rerank(query, candidates, top_k)
        _end = time.perf_counter()
        logger.info(f"Rerank耗时: {_end - _start:.4f}s")    
        
        if debug:
            debug_info['reranked_docs'] = reranked_docs

        _start = time.perf_counter()
        # --- D. Parent Expansion ---
        final_docs = self._expand_to_parent_context(reranked_docs)
        _end = time.perf_counter()
        logger.info(f"Parent Expansion耗时: {_end - _start:.4f}s")
        
        if debug:
            debug_info['final_docs'] = final_docs
            return debug_info
        return final_docs

    # -------------------------------------------------------------------------
    # 私有辅助方法
    # -------------------------------------------------------------------------

    def _search_single_channel(self, store, data, field, metric, search_params):
        try:
            res = store.client.search(
                collection_name=store.collection_name,
                data=data,
                anns_field=field,
                search_params={"metric_type": metric, "params": search_params},
                limit=3,
                output_fields=["text", "metadata", "source"]
            )
            
            docs = []
            for hits in res:
                for hit in hits:
                    # --- [修正] 单路检索解析逻辑 ---
                    entity = hit.get('entity', {})
                    
                    content = entity.get('text', "")
                    
                    # 获取嵌套的 metadata
                    meta = entity.get('metadata', {})
                    
                    # 获取分数 (单路检索通常返回 distance)
                    score = hit.get('score')
                    if score is None:
                        score = hit.get('distance')
                    score = score or 0.0
                    
                    doc = Document(page_content=content, metadata=meta)
                    doc.metadata["score"] = score
                    # 方便 debug 打印时看 header
                    doc.metadata["parent_header"] = meta.get("h2") or meta.get("h1") or "Unknown"
                    
                    docs.append(doc)
            return docs
            
        except Exception as e:
            logger.error(f"单路搜索失败 ({field}): {e}")
            return []

    def _rerank(self, query: str, docs: List[Document], top_k: int) -> List[Document]:
        if not docs or not TextReRank: return docs[:top_k]
        try:
            doc_texts = [d.page_content for d in docs]
            actual_top_k = min(len(docs), top_k)
            
            # Rerank
            logger.info(f"====Rerank parames: model={RERANK_MODEL} query={query} documents={doc_texts[:5]}, top_n={actual_top_k}")
            resp = TextReRank.call(
                model=RERANK_MODEL, 
                query=query, 
                documents=doc_texts,
                top_n=actual_top_k, 
                return_documents=False, 
                api_key=DASHSCOPE_API_KEY
            )
            
            ranked_docs = []
            filtered_count = 0
            if resp.output.results:
                for item in resp.output.results:
                    if item.relevance_score < _RERANK_SCORE_THRESHOLD:
                        filtered_count += 1
                        continue
                    original_doc = docs[item.index]
                    original_doc.metadata["rerank_score"] = item.relevance_score
                    ranked_docs.append(original_doc)
            if filtered_count:
                logger.info(
                    f"Rerank 阈值过滤: 丢弃 {filtered_count} 条低分文档 "
                    f"(threshold={_RERANK_SCORE_THRESHOLD}), 保留 {len(ranked_docs)} 条"
                )
            return ranked_docs
            
        except Exception as e:
            logger.error(traceback.format_exc())
            logger.warning(f"Rerank 失败，降级为原始顺序: {e}")
            return docs[:top_k]

    def _expand_to_parent_context(self, docs: List[Document]) -> List[Document]:
        expanded_docs = []
        for doc in docs:
            # 检查 metadata 中是否有 parent_context (在 Ingestion 阶段切分时存入)
            parent_context = doc.metadata.get("parent_context")
            
            if parent_context and len(parent_context) > len(doc.page_content):
                doc.metadata["original_child_content"] = doc.page_content 
                doc.page_content = parent_context # 替换为更完整的上下文
                doc.metadata["is_expanded"] = True
            else:
                doc.metadata["is_expanded"] = False
            
            expanded_docs.append(doc)
        return expanded_docs
    
# ==========================================
# 独立的展示工具函数 (保持不变，仅做微调适配)
# ==========================================
def print_retrieval_debug_info(debug_data: dict):
    print("\n" + "="*60)
    print("🕵️‍♀️ [DEBUG] 检索全链路分析报告 (Milvus 2.5 Server-side BM25)")
    print("="*60)

    # 1. 打印单路检索结果
    print("\n🔵 [Dense Only] 稠密检索 Top 3 (Embeddings):")
    if not debug_data.get('dense_only'): print("   (无结果)")
    for doc in debug_data.get('dense_only', []):

        header = doc.metadata.get('parent_header', 'N/A')
        # 【修复】使用 (val or 0) 防止 NoneType 报错
        score = doc.metadata.get('score') or 0 
        print(f"   - Score: {score:.4f} | Header: {header} | Content: {doc.page_content[:30]}...")

    print("\n🟣 [Sparse Only] 稀疏检索 Top 3 (Server-side BM25):")
    if not debug_data.get('sparse_only'): print("   (无结果)")
    for doc in debug_data.get('sparse_only', []):
        header = doc.metadata.get('parent_header', 'N/A')
        # 【修复】使用 (val or 0)
        score = doc.metadata.get('score') or 0
        print(f"   - Score: {score:.4f} | Header: {header} | Content: {doc.page_content[:30]}...")

    print("-" * 60)

    # 2. 打印 RRF 结果
    print("\n📌 [Stage 2] RRF 混合召回 (Top 5):")
    rrf_docs = debug_data.get('rrf_candidates', [])
    if not rrf_docs: print("   (无召回结果)")
    for i, doc in enumerate(rrf_docs):
        # 【修复】使用 (val or 0)
        score = doc.metadata.get("milvus_score") or 0
        header = doc.metadata.get("parent_header", "Unknown")
        print(f"   {i+1}. [RRF Score: {score:.4f}] [Header: {header}]")
        print(f"      📝 子块内容: {doc.page_content.replace(chr(10), ' ')[:50]}...")

    # 3. 打印 Rerank 结果
    print(f"\n📌 [Stage 3] Rerank 重排序 (Top {len(debug_data.get('reranked_docs', []))}):")
    reranked = debug_data.get('reranked_docs', [])
    for i, doc in enumerate(reranked):
        # 【修复】使用 (val or 0)
        score = doc.metadata.get("rerank_score") or 0
        header = doc.metadata.get("parent_header", "Unknown")
        print(f"   {i+1}. [Re-Score: {score:.4f}] [Header: {header}]")

    # 4. 打印最终结果
    print("\n📌 [Stage 4] 最终交付结果 (父块扩展后):")
    final_docs = debug_data.get('final_docs', [])
    for i, doc in enumerate(final_docs):
        is_parent = doc.metadata.get("is_expanded", False)
        type_icon = "🟢 父块 (Small-to-Big)" if is_parent else "⚪ 子块 (Raw)"
        header = doc.metadata.get("parent_header", "Unknown")
        # 【修复】使用 (val or 0)
        score = doc.metadata.get("rerank_score") or 0
        
        print(f"   {i+1}. {type_icon} [Score: {score:.4f}]")
        print(f"      🏷️ 来源标题: {header}")
        print(f"      📖 最终内容: {doc.page_content.replace(chr(10), ' ')[:100]}...")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    # 脚本模式：静默第三方库噪音，只保留本模块 INFO 日志
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.logging_config import setup_logging
    setup_logging(silence_libs=True, script_logger_name=__name__)

    # 确保 Ingestion 代码里使用的是 "default_rag_test_v1"
    collection_name = "default_rag_test_v1" 
    test_query = "退货政策是？" 

    print(f"\n🚀 开始测试 Retrieval 模块 (针对集合: {collection_name})...\n")

    retriever = Retrieval()
    
    try:
        result_data = retriever.retrieve(
            query=test_query, 
            collection_name=collection_name, 
            top_k=3, 
            debug=True 
        )

        if isinstance(result_data, dict):
            print_retrieval_debug_info(result_data)
        else:
            print("收到文档数量:", len(result_data))
            
    except Exception as e:
        print(f"❌ 测试运行失败: {e}")
        print("请检查：1. Milvus 是否运行 2. 集合名称是否正确 3. 是否已执行 Ingestion")