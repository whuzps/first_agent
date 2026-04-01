import hashlib
import logging
import os
import sys
import time
import traceback
from typing import Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from .document_loader import DocumentLoader
    from .model import Model
    from .milvus_store import MilvusStore
except ImportError:
    from document_loader import DocumentLoader
    from model import Model
    from milvus_store import MilvusStore

logger = logging.getLogger(__name__)

# 单批次 Embedding 请求的文本条数
_DEFAULT_EMBED_BATCH = int(os.getenv("RAG_EMBED_BATCH_SIZE", "50"))

# 单批次写入 Milvus 的行数（受 Milvus gRPC 消息大小限制，建议 ≤ 100）
_DEFAULT_INSERT_BATCH = int(os.getenv("RAG_INSERT_BATCH_SIZE", "50"))

# 每批次失败后的最大重试次数
_MAX_RETRY = int(os.getenv("RAG_BATCH_MAX_RETRY", "3"))

# 重试之间的等待秒数（指数退避）
_RETRY_BASE_WAIT = float(os.getenv("RAG_RETRY_BASE_WAIT", "2.0"))

# 语义去重：余弦相似度阈值（> 此值则视为重复，不入库）
_DEFAULT_SIM_THRESHOLD = float(os.getenv("RAG_DEDUP_SIMILARITY_THRESHOLD", "0.95"))


def _md5(text: str) -> str:
    """计算文本内容的 MD5 指纹（32 字符）"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


class Ingestion:
    def __init__(
        self,
        embed_batch_size: int = _DEFAULT_EMBED_BATCH,
        insert_batch_size: int = _DEFAULT_INSERT_BATCH,
        similarity_threshold: float = _DEFAULT_SIM_THRESHOLD,
    ):
        self.loader = DocumentLoader(child_chunk_size=500, child_overlap=50)
        self.dense_model = Model.get_dense_embedding_model()
        self.embed_batch = embed_batch_size
        self.insert_batch = insert_batch_size
        self.sim_threshold = similarity_threshold

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def ingest_file(self, file_path: str, collection_name: str) -> bool:
        """读取文件 → 流式切分 → 双重去重 → 分批向量化 → 写入 Milvus。

        去重策略（两层过滤，由粗到细）：
          Layer 1 — MD5 精确去重（零 API 开销）：
            同一批内先本地字典去重，再批量查 Milvus md5 索引，已存在则跳过 Embedding。
          Layer 2 — 向量语义去重（Embedding 后，写入前）：
            用稠密向量检索 top-1，余弦相似度 > threshold 则跳过写入。
        """
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return False

        file_size = os.path.getsize(file_path)
        logger.info(
            f"开始入库: {file_path}  大小={file_size / 1024:.1f} KB  "
            f"embed_batch={self.embed_batch}  insert_batch={self.insert_batch}  "
            f"sim_threshold={self.sim_threshold}"
        )

        store = MilvusStore(collection_name=collection_name)
        t_start = time.perf_counter()

        embed_buffer: list = []
        total_chunks = 0
        total_inserted = 0
        total_md5_dup = 0
        total_sim_dup = 0

        # 本地 MD5 集合：对同一批入库内的重复 chunk 也做去重
        seen_md5: set = set()

        for doc in self.loader.iter_chunks(file_path):
            total_chunks += 1

            # ── Layer 1-a: 本批次内 MD5 去重（纯本地，零开销）──
            doc_md5 = _md5(doc.page_content)
            if doc_md5 in seen_md5:
                total_md5_dup += 1
                continue
            seen_md5.add(doc_md5)

            doc.metadata["md5"] = doc_md5
            embed_buffer.append(doc)

            if len(embed_buffer) >= self.embed_batch:
                inserted, md5_dup, sim_dup = _flush(
                    store, self.dense_model, embed_buffer,
                    self.insert_batch, self.sim_threshold,
                )
                total_inserted += inserted
                total_md5_dup += md5_dup
                total_sim_dup += sim_dup
                embed_buffer.clear()

                if total_chunks % 500 == 0:
                    elapsed = time.perf_counter() - t_start
                    rate = total_chunks / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"进度: 已切分 {total_chunks} 片 | "
                        f"已入库 {total_inserted} 片 | "
                        f"MD5去重 {total_md5_dup} | 相似去重 {total_sim_dup} | "
                        f"速率 {rate:.1f} 片/s"
                    )

        # 处理尾部不足一批的数据
        if embed_buffer:
            inserted, md5_dup, sim_dup = _flush(
                store, self.dense_model, embed_buffer,
                self.insert_batch, self.sim_threshold,
            )
            total_inserted += inserted
            total_md5_dup += md5_dup
            total_sim_dup += sim_dup

        elapsed = time.perf_counter() - t_start
        total_skipped = total_md5_dup + total_sim_dup
        total_failed = total_chunks - total_inserted - total_skipped
        logger.info(
            f"入库完成: 总切片={total_chunks}  成功={total_inserted}  "
            f"MD5去重={total_md5_dup}  相似去重={total_sim_dup}  "
            f"失败={total_failed}  耗时={elapsed:.1f}s  "
            f"速率={total_chunks / elapsed:.1f} 片/s"
        )

        return total_inserted > 0


# ------------------------------------------------------------------
# 模块级辅助函数（ProcessPoolExecutor 子进程可直接调用）
# ------------------------------------------------------------------

def _flush(
    store: MilvusStore,
    dense_model,
    buf: list,
    insert_batch: int,
    sim_threshold: float,
) -> tuple:
    """对缓冲区执行 MD5 查重 → Embedding → 语义查重 → 写入。
    
    返回 (inserted, md5_dup_count, sim_dup_count)。
    """
    if not buf:
        return 0, 0, 0

    # ── Layer 1-b: 批量向 Milvus 查询已有 MD5 ──
    md5_list = [d.metadata["md5"] for d in buf]
    existing_md5 = store.exists_by_md5(md5_list)

    # 过滤掉库中已有的
    new_docs = []
    md5_dup = 0
    for d in buf:
        if d.metadata["md5"] in existing_md5:
            md5_dup += 1
        else:
            new_docs.append(d)

    if md5_dup:
        logger.info(f"MD5 去重: 本批 {len(buf)} 条，跳过 {md5_dup} 条已有")

    if not new_docs:
        return 0, md5_dup, 0

    # ── Embedding（仅对真正需要入库的 chunk 计算向量）──
    inserted = 0
    sim_dup = 0
    for k in range(0, len(new_docs), insert_batch):
        sub = new_docs[k: k + insert_batch]
        texts = [d.page_content for d in sub]
        _inserted, _sim_dup = _embed_dedup_insert(
            store, dense_model, sub, texts, sim_threshold
        )
        inserted += _inserted
        sim_dup += _sim_dup

    return inserted, md5_dup, sim_dup


def _embed_dedup_insert(
    store: MilvusStore,
    dense_model,
    docs: list,
    texts: list,
    sim_threshold: float,
) -> tuple:
    """单子批：计算向量 → 逐条语义查重 → 写入 Milvus。
    
    返回 (inserted_count, sim_dup_count)。
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, _MAX_RETRY + 1):
        try:
            dense_vectors = dense_model.embed_documents(texts)

            # ── Layer 2: 逐条语义去重（向量检索 top-1，比较余弦相似度）──
            data_rows = []
            sim_dup = 0
            for j, doc in enumerate(docs):
                vec = dense_vectors[j]

                if sim_threshold < 1.0:
                    hits = store.find_similar(vec, top_k=1)
                    if hits and hits[0]["distance"] > sim_threshold:
                        sim_dup += 1
                        logger.debug(
                            f"语义去重: sim={hits[0]['distance']:.4f} > {sim_threshold}  "
                            f"text={doc.page_content[:60]}…"
                        )
                        continue

                data_rows.append({
                    "text":         doc.page_content,
                    "md5":          doc.metadata.get("md5", _md5(doc.page_content)),
                    "source":       doc.metadata.get("source", ""),
                    "metadata":     doc.metadata,
                    "dense_vector": vec,
                })

            if data_rows:
                store.insert(data_rows)
            return len(data_rows), sim_dup

        except Exception as exc:
            last_exc = exc
            wait = _RETRY_BASE_WAIT * (2 ** (attempt - 1))
            logger.warning(
                f"批次写入失败 (第 {attempt}/{_MAX_RETRY} 次): {exc}  "
                f"{wait:.0f}s 后重试…"
            )
            time.sleep(wait)

    logger.error(f"批次写入放弃（已重试 {_MAX_RETRY} 次）: {last_exc}")
    logger.debug(traceback.format_exc())
    return 0, 0


if __name__ == "__main__":
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.logging_config import setup_logging
    setup_logging()
    ingestion = Ingestion()
    path = "" # 知识库路径
    ingestion.ingest_file(path, "default_rag_test_v1",)
