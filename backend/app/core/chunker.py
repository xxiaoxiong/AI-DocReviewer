"""
智能分块器 - 解决语义断裂问题
"""
from typing import List, Dict, Any, Optional
from loguru import logger

from ..models.document import DocumentChunk
from ..config import settings


class SmartChunker:
    """
    智能分块器
    
    核心策略：
    1. 按段落分块（保持语义完整）
    2. 添加上下文摘要（解决跨段落问题）
    3. 滑动窗口（处理长段落）
    """
    
    def __init__(
        self,
        chunk_size: int = None,
        overlap: int = None
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.overlap = overlap or settings.chunk_overlap
    
    def chunk_by_paragraphs(
        self,
        document_structure: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """
        按段落分块（推荐方案）
        
        优势：
        - 保持段落完整性
        - 不会切断句子
        - 保留章节信息
        
        Args:
            document_structure: 文档结构（来自 DocumentParser）
        
        Returns:
            文档块列表
        """
        chunks = []
        paragraphs = document_structure.get("paragraphs", [])
        
        for i, para in enumerate(paragraphs):
            # 获取前后文摘要
            context_before = self._get_context_summary(paragraphs, i, direction="before")
            context_after = self._get_context_summary(paragraphs, i, direction="after")
            
            # 如果段落太长，需要进一步切分
            if len(para["text"]) > self.chunk_size:
                sub_chunks = self._split_long_paragraph(para, i)
                chunks.extend(sub_chunks)
            else:
                chunk = DocumentChunk(
                    chunk_id=f"chunk_{i}",
                    text=para["text"],
                    start_pos=0,  # 可以根据实际需求计算
                    end_pos=len(para["text"]),
                    page=None,  # Word 文档没有页码概念
                    section=para.get("section"),
                    context_before=context_before,
                    context_after=context_after
                )
                chunks.append(chunk)
        
        logger.info(f"文档分块完成: {len(chunks)} 个块")
        return chunks
    
    def chunk_by_sections(
        self,
        document_structure: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """
        按章节分块（适合结构化文档）
        
        Args:
            document_structure: 文档结构
        
        Returns:
            文档块列表
        """
        chunks = []
        sections = document_structure.get("sections", [])
        
        for i, section in enumerate(sections):
            # 合并章节内的所有段落
            section_text = "\n".join([
                p["text"] for p in section.get("paragraphs", [])
            ])
            
            if not section_text:
                continue
            
            # 如果章节太长，按段落切分
            if len(section_text) > self.chunk_size * 2:
                para_chunks = self.chunk_by_paragraphs({
                    "paragraphs": section["paragraphs"]
                })
                chunks.extend(para_chunks)
            else:
                chunk = DocumentChunk(
                    chunk_id=f"section_{i}",
                    text=section_text,
                    start_pos=0,
                    end_pos=len(section_text),
                    section=section.get("title"),
                    context_before=None,
                    context_after=None
                )
                chunks.append(chunk)
        
        return chunks
    
    def _get_context_summary(
        self,
        paragraphs: List[Dict],
        current_index: int,
        direction: str = "before",
        window_size: int = 2
    ) -> Optional[str]:
        """
        获取上下文摘要
        
        这是解决语义断裂的关键！
        
        Args:
            paragraphs: 所有段落
            current_index: 当前段落索引
            direction: "before" 或 "after"
            window_size: 窗口大小（前后各取几个段落）
        
        Returns:
            上下文摘要
        """
        if direction == "before":
            start = max(0, current_index - window_size)
            end = current_index
            context_paras = paragraphs[start:end]
        else:  # after
            start = current_index + 1
            end = min(len(paragraphs), current_index + 1 + window_size)
            context_paras = paragraphs[start:end]
        
        if not context_paras:
            return None
        
        # 提取摘要（取每段前50字）
        summaries = []
        for p in context_paras:
            text = p["text"]
            summary = text[:50] + "..." if len(text) > 50 else text
            summaries.append(summary)
        
        return " | ".join(summaries)
    
    def _split_long_paragraph(
        self,
        para: Dict[str, Any],
        para_index: int
    ) -> List[DocumentChunk]:
        """
        切分长段落（使用滑动窗口）
        
        Args:
            para: 段落数据
            para_index: 段落索引
        
        Returns:
            子块列表
        """
        text = para["text"]
        chunks = []
        start = 0
        sub_index = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # 尝试在句号处断开
            if end < len(text):
                last_period = chunk_text.rfind("。")
                if last_period > self.chunk_size * 0.5:  # 至少保留一半
                    end = start + last_period + 1
                    chunk_text = text[start:end]
            
            chunk = DocumentChunk(
                chunk_id=f"chunk_{para_index}_{sub_index}",
                text=chunk_text,
                start_pos=start,
                end_pos=end,
                section=para.get("section"),
                context_before=None,
                context_after=None
            )
            chunks.append(chunk)
            
            # 滑动窗口（有重叠）
            start += (self.chunk_size - self.overlap)
            sub_index += 1
        
        return chunks
    
    def merge_small_chunks(
        self,
        chunks: List[DocumentChunk],
        min_size: int = 100
    ) -> List[DocumentChunk]:
        """
        合并过小的块
        
        Args:
            chunks: 原始块列表
            min_size: 最小块大小
        
        Returns:
            合并后的块列表
        """
        merged = []
        buffer = []
        
        for chunk in chunks:
            if len(chunk.text) < min_size:
                buffer.append(chunk)
            else:
                if buffer:
                    # 合并 buffer 中的小块
                    merged_text = " ".join([c.text for c in buffer])
                    merged_chunk = DocumentChunk(
                        chunk_id=f"merged_{len(merged)}",
                        text=merged_text,
                        start_pos=buffer[0].start_pos,
                        end_pos=buffer[-1].end_pos,
                        section=buffer[0].section
                    )
                    merged.append(merged_chunk)
                    buffer = []
                
                merged.append(chunk)
        
        # 处理剩余的 buffer
        if buffer:
            merged_text = " ".join([c.text for c in buffer])
            merged_chunk = DocumentChunk(
                chunk_id=f"merged_{len(merged)}",
                text=merged_text,
                start_pos=buffer[0].start_pos,
                end_pos=buffer[-1].end_pos,
                section=buffer[0].section
            )
            merged.append(merged_chunk)
        
        return merged

