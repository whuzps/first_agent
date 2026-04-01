import logging
import traceback
from pymilvus import (
    MilvusClient, 
    DataType, 
    Function, 
    FunctionType, 
    AnnSearchRequest, 
    RRFRanker,
    WeightedRanker
)
try:
    from service.config import COLLECTION_NAME, MILVUS_HOST, MILVUS_PORT, MILVUS_TOKEN, EMBEDDING_DIM
except Exception:
    from core.config import COLLECTION_NAME, MILVUS_HOST, MILVUS_PORT, MILVUS_TOKEN, EMBEDDING_DIM

logger = logging.getLogger(__name__)

class MilvusStore:
    def __init__(self, collection_name: str = None):
        self.collection_name = collection_name or COLLECTION_NAME
        self.uri = f"http://{MILVUS_HOST}:{MILVUS_PORT}"
        
        # 1. 初始化客户端 (Milvus 2.5 推荐方式)
        self.client = MilvusClient(uri=self.uri, token=MILVUS_TOKEN if MILVUS_TOKEN else "")
        
        # 2. 初始化集合
        self._init_collection()

    def _init_collection(self):
        """
        初始化逻辑：检查集合是否存在，不存在则创建（含 BM25 Function 定义）
        """
        if self.client.has_collection(self.collection_name):
            logger.info(f"加载现有集合: {self.collection_name}")
            self.client.load_collection(self.collection_name)
        else:
            logger.info(f"集合不存在，正在创建: {self.collection_name}")
            self._create_collection_v25()

    def _create_collection_v25(self):
        """
        创建带有 Server-side BM25 功能的 Schema（Milvus 2.5+ 特性）
        """
        # 1. 创建 Schema (开启动态字段以存储 metadata)
        schema = MilvusClient.create_schema(
            enable_dynamic_field=True,
            description="Hybrid RAG (Dense + Server-side BM25)"
        )

        # 2. 配置分析器 (Analyzer)
        # 注意：处理中文推荐配置 "type": "chinese" (需要 Milvus 编译了支持库)
        # 如果是标准 Docker 镜像，默认可用 "english" 或 "standard"
        analyzer_params = {"type": "chinese"} 

        # 3. 添加字段
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)

        # chunk 内容的 MD5 指纹（用于精确去重，32 字符固定长度）
        schema.add_field(field_name="md5", datatype=DataType.VARCHAR, max_length=32)
        
        # [关键] text 字段必须开启 analyzer，用于自动生成稀疏向量
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535, 
                         enable_analyzer=True, analyzer_params=analyzer_params, enable_match=True)
        
        # [关键] 稀疏向量字段
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        
        # 稠密向量字段
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)

        # 4. [核心特性] 添加 BM25 Function
        # 将 text 列映射到 sparse_vector 列
        bm25_function = Function(
            name="text_bm25_mapper",
            function_type=FunctionType.BM25,
            input_field_names=["text"],
            output_field_names="sparse_vector",
        )
        schema.add_function(bm25_function)

        # 5. 配置索引
        index_params = self.client.prepare_index_params()

        # md5 标量索引（用于精确去重过滤，INVERTED 索引适合等值查询）
        index_params.add_index(
            field_name="md5",
            index_name="md5_idx",
            index_type="INVERTED",
        )

        # 稠密索引
        # 可选参数 params={"M": 16, "efConstruction": 200}
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_idx",
            index_type="HNSW",
            metric_type="COSINE"
        )

        # 稀疏索引 ( 2.6 使用索引 SPARSE_INVERTED_INDEX)
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_idx",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25"
        )

        # 6. 创建集合
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )
        logger.info(f"集合 {self.collection_name} 创建完成 (Milvus 2.5 Schema)")

    def insert(self, data_dict_list: list):
        """
        插入数据
        注意：data_dict_list 中只需包含 'text' 和 'dense_vector'，
        'sparse_vector' 会由 Milvus 服务端自动生成。
        """
        try:
            res = self.client.insert(
                collection_name=self.collection_name,
                data=data_dict_list
            )
            logger.info(f"成功插入 {res['insert_count']} 条数据")
            return res
        except Exception as e:
            logger.error(f"插入数据失败: {e}")
            raise e

    # ------------------------------------------------------------------
    # 去重查询
    # ------------------------------------------------------------------

    def exists_by_md5(self, md5_list: list[str]) -> set[str]:
        """批量查询已存在的 MD5，返回库中已有的 MD5 集合（用于精确去重）。
        
        利用 md5 字段的 INVERTED 索引做等值过滤，仅查主键，开销极低。
        """
        if not md5_list:
            return set()
        try:
            # 构造 IN 表达式："md5 in ['abc123', 'def456', ...]"
            quoted = ", ".join(f'"{m}"' for m in md5_list)
            expr = f"md5 in [{quoted}]"
            results = self.client.query(
                collection_name=self.collection_name,
                filter=expr,
                output_fields=["md5"],
                limit=len(md5_list),
            )
            return {r["md5"] for r in results}
        except Exception as e:
            logger.warning(f"MD5 去重查询失败，跳过去重: {e}")
            return set()

    def find_similar(self, dense_vector: list, top_k: int = 1) -> list[dict]:
        """单向量相似度检索，返回 top_k 条结果（含 distance/score）。
        
        用于语义去重：COSINE 距离值范围 [0, 2]，score = 1 - distance/2 ∈ [0,1]。
        Milvus COSINE metric 直接返回余弦相似度（越大越相似）。
        """
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                data=[dense_vector],
                anns_field="dense_vector",
                search_params={"metric_type": "COSINE"},
                limit=top_k,
                output_fields=["text", "md5"],
            )
            # results[0] 对应第一条 query 的命中列表
            hits = []
            for hit in results[0] if results else []:
                hits.append({
                    "id": hit["id"],
                    "distance": hit["distance"],   # COSINE metric: 即余弦相似度
                    "text": hit["entity"].get("text", ""),
                    "md5": hit["entity"].get("md5", ""),
                })
            return hits
        except Exception as e:
            logger.error(f"相似度查重检索失败，跳过去重: {traceback.format_exc()}")
            # logger.warning(f"相似度查重检索失败，跳过去重: {e}")
            return []

    def hybrid_search(self, query_text: str, query_dense_vector: list, k: int = 5, ranker_params: dict = None):
        """
        [更新] 混合检索
        参数由 (dense, sparse) 变为 (text, dense)。
        因为 Milvus 服务端知道如何将 query_text 转换为稀疏向量用于检索。
        
        Args:
            query_text: 查询文本（用于 BM25 稀疏检索）
            query_dense_vector: 查询稠密向量
            k: 返回结果数量
            ranker_params: 排名器参数，格式：
                - {"strategy": "weighted", "weights": [0.6, 0.4]}
                - {"strategy": "rrf", "k": 60}
                默认为 {"strategy": "weighted", "weights": [0.6, 0.4]}
        """
        # 1. 稠密向量检索请求
        dense_req = AnnSearchRequest(
            data=[query_dense_vector],
            anns_field="dense_vector",
            param={"metric_type": "COSINE"},
            limit=k * 2
        )

        # 2. 稀疏向量检索请求 (BM25)
        # [Milvus 2.5 特性] 直接传入文本，Milvus 会利用 Schema 中的 Function 自动转换
        sparse_req = AnnSearchRequest(
            data=[query_text],  # 直接传字符串列表
            anns_field="sparse_vector",
            param={"metric_type": "BM25", "params": {"drop_ratio_search": 0.2}}, # BM25 Metric
            limit=k * 2
        )

        # 3. 根据 ranker_params 选择排名器
        try:
            # 设置默认排名器参数
            if ranker_params is None:
                ranker_params = {
                    "strategy": "weighted",
                    "weights": [0.6, 0.4]
                }
            
            strategy = ranker_params.get("strategy", "weighted").lower()
            logger.info(f"[混合检索] 使用排名策略: {strategy}")
            
            # 根据 strategy 选择排名器
            if strategy == "weighted":
                weights = ranker_params.get("weights", [0.6, 0.4])
                ranker = WeightedRanker(*weights)
                logger.info(f"[混合检索] WeightedRanker 权重: {weights}")
            elif strategy == "rrf":
                rrf_k = ranker_params.get("k", 60)
                ranker = RRFRanker(k=rrf_k)
                logger.info(f"[混合检索] RRFRanker k值: {rrf_k}")
            else:
                logger.warning(f"[混合检索] 未知策略 '{strategy}'，使用默认 weighted")
                ranker = WeightedRanker(0.6, 0.4)
            
            # 执行混合检索
            results = self.client.hybrid_search(
                collection_name=self.collection_name,
                reqs=[dense_req, sparse_req],
                ranker=ranker,
                limit=k,
                output_fields=["text", "source", "metadata"] # 返回字段
            )
            return results
        except Exception as e:
            logger.error(f"混合检索失败: {e}")
            return []

    def clear(self):
        """清空并重建集合"""
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
            logger.info(f"集合 {self.collection_name} 已删除")
            # self._create_collection_v25()

    def drop(self):
        """彻底删除集合（不重建）"""
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
            logger.info(f"集合 {self.collection_name} 已彻底删除")

    # ------------------------------------------------------------------
    # 管理接口
    # ------------------------------------------------------------------

    def list_collections(self) -> list[str]:
        """列出所有集合名称"""
        try:
            return self.client.list_collections()
        except Exception as e:
            logger.error(f"列出集合失败: {e}")
            return []

    def get_collection_stats(self, name: str | None = None) -> dict:
        """获取指定集合的实时统计信息。
        
        先 flush 再读取 row_count，确保包含最近插入但尚未落盘的数据。
        """
        col = name or self.collection_name
        try:
            # flush 强制将内存中的 segment 写入磁盘，使 row_count 反映真实最新值
            self.client.flush(col)
            stats = self.client.get_collection_stats(col)
            row_count = int(stats.get("row_count", 0))
            return {
                "collection_name": col,
                "row_count": row_count,
                "exists": True,
            }
        except Exception as e:
            logger.warning(f"获取集合统计失败 [{col}]: {e}")
            return {
                "collection_name": col,
                "row_count": 0,
                "exists": self.client.has_collection(col) if col else False,
            }


if __name__ == "__main__":
    store = MilvusStore(collection_name="default_rag_test_v1")
    store.clear()
