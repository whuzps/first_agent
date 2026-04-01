import os
import re
import uuid
import logging
from typing import List, Optional, Generator, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader

logger = logging.getLogger(__name__)

# parent_context 截断阈值：存入 Milvus metadata 的最大字符数
# Milvus VARCHAR(65535) 限制，metadata 字段还要留空间给其他字段，保守取 4000
_MAX_PARENT_CONTEXT = int(os.getenv("RAG_MAX_PARENT_CONTEXT", "4000"))

# 超过此大小（字节）的文件走流式切分，避免一次性全量加载
_LARGE_FILE_THRESHOLD = int(os.getenv("RAG_LARGE_FILE_THRESHOLD", str(20 * 1024 * 1024)))  # 20 MB

# separators=["\n\n", "\n", "。", "！", " ", "？"],

class DocumentLoader:
    def __init__(self, child_chunk_size: int = 500, child_overlap: int = 50):
        self.child_chunk_size = child_chunk_size
        self.child_overlap = child_overlap
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_overlap,
            separators=[
                "\n\nQ：",
                "\n\nA：",
                "\n\n问题：",
                "\n\n答案：",
                "\n\n",
                "\n",
            ],
            keep_separator=True,
        )

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def load_and_split(self, file_path: str) -> List[Document]:
        """一次性加载并切分（适合中等大小文件，< 20 MB）。"""
        return list(self.iter_chunks(file_path))

    def iter_chunks(self, file_path: str) -> Generator[Document, None, None]:
        """流式生成 Document 切片，大文件首选，避免峰值内存爆炸。"""
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return

        ext = os.path.splitext(file_path)[1].lower()
        file_size = os.path.getsize(file_path)
        base_meta = {"source": os.path.basename(file_path)}
        is_large = file_size > _LARGE_FILE_THRESHOLD

        if is_large:
            logger.info(f"大文件模式: {file_path} ({file_size / 1024 / 1024:.1f} MB)")

        if ext == ".md":
            # Markdown 必须全量解析标题层级，但 parent_context 会截断
            text = self._load_full_text(file_path)
            if text:
                yield from self._process_markdown_recursive(text, base_meta)
        elif ext == ".pdf":
            yield from self._iter_pdf(file_path, base_meta)
        else:
            # TXT / DOCX 等——大文件走流式分段，小文件走全量
            if is_large and ext == ".txt":
                yield from self._iter_txt_streaming(file_path, base_meta)
            else:
                text = self._load_full_text(file_path)
                if text:
                    yield from self._process_generic_text(text, base_meta)

    # ------------------------------------------------------------------
    # 内部方法：各格式处理
    # ------------------------------------------------------------------

    def _process_markdown_recursive(
        self, text: str, base_meta: dict
    ) -> Generator[Document, None, None]:
        """
        递归层级切分（H1 → H2 → H3 → 500字子块）。
        parent_context 截断至 _MAX_PARENT_CONTEXT 字符，防止 Milvus 写入超限。
        """
        h1_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1")], strip_headers=False
        )
        h1_docs = h1_splitter.split_text(text)

        for h1_doc in h1_docs:
            h1_content = h1_doc.page_content
            # 截断后用于 parent_context 存储
            h1_context_stored = _truncate(h1_content, _MAX_PARENT_CONTEXT)

            h2_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=[("##", "h2")], strip_headers=False
            )
            h2_docs = h2_splitter.split_text(h1_content)

            for h2_doc in h2_docs:
                h2_content = h2_doc.page_content
                is_h2_block = "h2" in h2_doc.metadata
                h2_context_stored = _truncate(h2_content, _MAX_PARENT_CONTEXT)

                # H2 块的父上下文是 H1；H1 前言块无父
                parent_for_h2 = h1_context_stored if is_h2_block else ""

                h3_splitter = MarkdownHeaderTextSplitter(
                    headers_to_split_on=[("###", "h3")], strip_headers=False
                )
                h3_docs = h3_splitter.split_text(h2_content)

                for h3_doc in h3_docs:
                    h3_content = h3_doc.page_content
                    is_h3_block = "h3" in h3_doc.metadata

                    # H3 块父上下文是 H2；H2 intro 继承上一层
                    if is_h3_block:
                        real_parent_context = h2_context_stored
                    else:
                        real_parent_context = parent_for_h2

                    leaf_parent_id = str(uuid.uuid4())
                    for chunk_text in self.child_splitter.split_text(h3_content):
                        yield Document(
                            page_content=chunk_text,
                            metadata={
                                **base_meta,
                                **h1_doc.metadata,
                                **h2_doc.metadata,
                                **h3_doc.metadata,
                                "parent_id": leaf_parent_id,
                                "parent_context": real_parent_context,
                            },
                        )

    def _process_generic_text(
        self, text: str, base_meta: dict
    ) -> Generator[Document, None, None]:
        """通用文本（TXT/DOCX）的切分，自动检测 QA 格式。

        如果文本包含 Q：/A：结构，使用 QA 感知切分：
          - 先按 QA 对拆分
          - 短于 chunk_size 的 QA 对直接入库
          - 超长答案在答案内部按段落二次切分，每个子 chunk 头部拼回问题
        非 QA 格式文本走通用 RecursiveCharacterTextSplitter。
        """
        if _is_qa_format(text):
            yield from self._split_qa(text, base_meta)
        else:
            for doc in self.child_splitter.create_documents([text], metadatas=[base_meta]):
                logger.info(f"切片 chunk content: {doc.page_content} \n")
                yield doc

    def _split_qa(
        self, text: str, base_meta: dict
    ) -> Generator[Document, None, None]:
        """QA 感知切分：问题必须完整保留在每个 chunk 头部。"""
        qa_pairs = _extract_qa_pairs(text)
        logger.info(f"QA 切分: 共识别 {len(qa_pairs)} 个问答对")

        # 用于超长答案二次切分的 splitter（按段落/换行/句号切分，不含 QA 分隔符）
        answer_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_chunk_size,
            chunk_overlap=self.child_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", " "],
            keep_separator=True,
        )

        for question, answer, tag in qa_pairs:
            full_qa = f"{question}\n{answer}"
            if tag:
                full_qa += f"\n{tag}"

            # 问题 + 答案整体不超过 chunk_size → 整块入库
            if len(full_qa) <= self.child_chunk_size:
                yield Document(page_content=full_qa, metadata={**base_meta, "qa_question": question})
                continue

            # 超长答案 → 对答案部分做二次切分
            # 预留问题长度 + 换行，确保拼接后每个 chunk 不超限
            q_prefix = question + "\n"
            tag_suffix = f"\n{tag}" if tag else ""
            available = self.child_chunk_size - len(q_prefix) - len(tag_suffix)
            if available < 100:
                # 极端情况：问题本身就超长，退化为直接切分
                available = self.child_chunk_size

            sub_splitter = RecursiveCharacterTextSplitter(
                chunk_size=available,
                chunk_overlap=min(self.child_overlap, available // 4),
                separators=["\n\n", "\n", "。", "！", "？", "；", " "],
                keep_separator=True,
            )
            answer_chunks = sub_splitter.split_text(answer)

            for chunk in answer_chunks:
                chunk_text = f"{q_prefix}{chunk.strip()}"
                if tag_suffix:
                    chunk_text += tag_suffix
                yield Document(
                    page_content=chunk_text,
                    metadata={**base_meta, "qa_question": question},
                )

    def _iter_txt_streaming(
        self, file_path: str, base_meta: dict, read_block: int = 2 * 1024 * 1024
    ) -> Generator[Document, None, None]:
        """
        超大 TXT 流式切分：每次读 2 MB 滑动窗口，避免全文入内存。
        overlap = child_overlap 字符（跨块边界保持上下文连贯性）。
        """
        overlap_text = ""
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            while True:
                block = f.read(read_block)
                if not block:
                    break
                segment = overlap_text + block
                chunks = self.child_splitter.split_text(segment)
                # 最后一块可能不完整，留作下次的 overlap
                if len(chunks) > 1:
                    for c in chunks[:-1]:
                        yield Document(page_content=c, metadata=base_meta)
                    # 保留最后一块（可能是不完整的段落）
                    overlap_text = chunks[-1]
                else:
                    # 块太短，全部作为 overlap 留到下次
                    overlap_text = segment
            # 处理最后的残余
            if overlap_text.strip():
                for c in self.child_splitter.split_text(overlap_text):
                    yield Document(page_content=c, metadata=base_meta)

    def _iter_pdf(
        self, file_path: str, base_meta: dict
    ) -> Generator[Document, None, None]:
        """
        PDF 按页流式处理：每页单独切分，避免一次性加载所有页。
        PyPDFLoader 内部已支持按页，这里逐页处理。
        """
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(file_path)
            # lazy_load() 按页惰性加载，不全部入内存
            for page_doc in loader.lazy_load():
                page_meta = {**base_meta, "page": page_doc.metadata.get("page", 0)}
                for chunk_text in self.child_splitter.split_text(page_doc.page_content):
                    yield Document(page_content=chunk_text, metadata=page_meta)
        except Exception as e:
            logger.error(f"PDF 流式加载失败，降级全量加载: {e}")
            text = self._load_full_text(file_path)
            if text:
                yield from self._process_generic_text(text, base_meta)

    def _load_full_text(self, file_path: str) -> str:
        """全量加载文件文本（仅小文件 / MD 使用）。"""
        loader = self._select_loader(file_path)
        if not loader:
            return ""
        docs = loader.load()
        return "\n\n".join(d.page_content for d in docs)

    def _select_loader(self, file_path: str):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return PyPDFLoader(file_path)
        if ext == ".docx":
            return Docx2txtLoader(file_path)
        return TextLoader(file_path, autodetect_encoding=True)


# ------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------

def _truncate(text: str, max_len: int) -> str:
    """截断文本，超过 max_len 时保留前段并追加省略标记。"""
    if not text or len(text) <= max_len:
        return text
    return text[:max_len] + "…（内容过长已截断）"


# QA 格式检测：至少存在 3 个 Q：/A：配对，才认定为 QA 格式
_QA_HEAD_RE = re.compile(r"^Q[：:]", re.MULTILINE)


def _is_qa_format(text: str) -> bool:
    """判断文本是否为 Q/A 问答格式"""
    return len(_QA_HEAD_RE.findall(text)) >= 3


# QA 对提取：按 "Q：" 开头拆分，再从每对中分离 Q/A/标签
_QA_SPLIT_RE = re.compile(r"(?=\nQ[：:]|\AQ[：:])", re.MULTILINE)
_TAG_RE = re.compile(r"\n(标签[：:].*)$")


def _extract_qa_pairs(text: str) -> List[Tuple[str, str, str]]:
    """从 QA 格式文本中提取 (question, answer, tag) 三元组列表。

    规则：
      - 以 Q：开头到下一个 Q：之间为一个完整 QA 块
      - 块内第一个 A：分割问题和答案
      - 末尾「标签：xxx」提取为 tag
    """
    blocks = _QA_SPLIT_RE.split(text)
    pairs = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # 分离标签
        tag = ""
        tag_match = _TAG_RE.search(block)
        if tag_match:
            tag = tag_match.group(1).strip()
            block = block[:tag_match.start()].strip()

        # 按第一个 A：分割
        a_pos = re.search(r"\nA[：:]", block)
        if a_pos:
            question = block[:a_pos.start()].strip()
            answer = block[a_pos.start():].strip()
        else:
            question = block
            answer = ""

        if question:
            pairs.append((question, answer, tag))

    return pairs
