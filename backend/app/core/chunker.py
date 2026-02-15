"""
智能分块器 - 解决语义断裂问题（深度优化版）
"""
from typing import List, Dict, Any, Optional
from loguru import logger

from ..models.document import DocumentChunk
from ..config import settings


class SmartChunker:
    """
    智能分块器（深度优化版）
    
    核心策略：
    1. 按段落分块（保持语义完整）
    2. 过滤垃圾块（太短、无意义的内容）
    3. 智能合并小块（保证最小语义单元）
    4. 添加上下文摘要（解决跨段落问题）
    5. 滑动窗口（处理长段落）
    
    优化点：
    - 最小块大小：30字符（避免垃圾块）
    - 智能合并：相邻小块自动合并
    - 标题识别：标题与正文合并
    - 空行过滤：自动跳过空段落
    """
    
    def __init__(
        self,
        chunk_size: int = None,
        overlap: int = None,
        min_chunk_size: int = 20,  # 最小块大小（字符数）- 过滤垃圾块
        merge_threshold: int = 50   # 合并阈值 - 只合并非常小的块
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.overlap = overlap or settings.chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.merge_threshold = merge_threshold
    
    def chunk_by_paragraphs(
        self,
        document_structure: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """
        按段落分块（深度优化版）
        
        优化策略：
        1. 过滤空段落和垃圾段落
        2. 识别标题并与后续内容合并
        3. 合并过小的段落
        4. 保持语义完整性
        
        Args:
            document_structure: 文档结构（来自 DocumentParser）
        
        Returns:
            文档块列表
        """
        paragraphs = document_structure.get("paragraphs", [])
        
        # 第一步：预处理 - 过滤和清洗
        cleaned_paragraphs = self._preprocess_paragraphs(paragraphs)
        
        if not cleaned_paragraphs:
            logger.warning("文档没有有效段落")
            return []
        
        # 第二步：初步分块
        raw_chunks = []
        for i, para in enumerate(cleaned_paragraphs):
            # 如果段落太长，需要进一步切分
            if len(para["text"]) > self.chunk_size:
                sub_chunks = self._split_long_paragraph(para, i)
                raw_chunks.extend(sub_chunks)
            else:
                chunk = DocumentChunk(
                    chunk_id=f"chunk_{i}",
                    text=para["text"],
                    start_pos=0,
                    end_pos=len(para["text"]),
                    page=None,
                    section=para.get("section"),
                    context_before=None,
                    context_after=None
                )
                raw_chunks.append(chunk)
        
        # 第三步：智能合并小块
        merged_chunks = self._smart_merge_chunks(raw_chunks)
        
        # 第四步：添加上下文摘要
        final_chunks = self._add_context_summary(merged_chunks)
        
        # 第五步：重新编号
        for i, chunk in enumerate(final_chunks):
            chunk.chunk_id = f"chunk_{i}"
        
        logger.info(f"文档分块完成: {len(paragraphs)} 个原始段落 -> {len(final_chunks)} 个有效块")
        logger.info(f"   过滤了 {len(paragraphs) - len(cleaned_paragraphs)} 个垃圾段落")
        logger.info(f"   合并了 {len(raw_chunks) - len(merged_chunks)} 个小块")
        
        return final_chunks
    
    def _preprocess_paragraphs(self, paragraphs: List[Dict]) -> List[Dict]:
        """
        预处理段落：过滤垃圾、清洗数据
        
        过滤规则（更宽松）：
        1. 空段落
        2. 只有标点符号的段落
        3. 长度小于10字符且不是标题的段落
        4. 只有数字的段落
        
        Args:
            paragraphs: 原始段落列表
        
        Returns:
            清洗后的段落列表
        """
        cleaned = []
        
        for para in paragraphs:
            text = para["text"].strip()
            
            # 跳过空段落
            if not text:
                continue
            
            # 跳过只有标点符号的段落
            if all(c in '()（）[]【】{}「」『』<>《》、，。；：！？\n\t ' for c in text):
                logger.debug(f"过滤垃圾段落（只有标点）: {text}")
                continue
            
            # 跳过只有数字的段落
            if text.replace(' ', '').replace('\t', '').isdigit():
                logger.debug(f"过滤垃圾段落（只有数字）: {text}")
                continue
            
            # 检查是否是标题（标题可以短一些）
            is_heading = self._is_likely_heading(text, para.get("style", ""))
            
            # 只过滤非常短的非标题段落（<10字符）
            if not is_heading and len(text) < 10:
                logger.debug(f"过滤垃圾段落（太短）: {text} ({len(text)}字符)")
                continue
            
            cleaned.append(para)
        
        return cleaned
    
    def _is_likely_heading(self, text: str, style: str) -> bool:
        """
        判断是否可能是标题
        
        Args:
            text: 段落文本
            style: 段落样式
        
        Returns:
            是否是标题
        """
        # 样式判断
        if 'heading' in style.lower() or 'title' in style.lower():
            return True
        
        # 长度判断（标题通常较短）
        if len(text) > 50:
            return False
        
        # 模式判断
        import re
        
        # 中文数字标题：一、二、三
        if re.match(r'^[一二三四五六七八九十]+[、．]', text):
            return True
        
        # 阿拉伯数字标题：1. 2. 3.
        if re.match(r'^\d+[\.\．、]', text):
            return True
        
        # 章节标题：第X章
        if re.match(r'^第[一二三四五六七八九十\d]+[章节部分]', text):
            return True
        
        return False
    
    def _smart_merge_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        智能合并小块（优化版 - 避免过度合并）
        
        策略：
        1. 只合并非常小的块（<50字符）
        2. 合并后不超过最大块大小的70%
        3. 保持段落独立性
        4. 标题单独成块或与紧邻的小段落合并
        
        Args:
            chunks: 原始块列表
        
        Returns:
            合并后的块列表
        """
        if not chunks:
            return []
        
        merged = []
        buffer = []
        max_merge_size = int(self.chunk_size * 0.7)  # 合并后不超过70%的chunk_size
        
        for i, chunk in enumerate(chunks):
            chunk_len = len(chunk.text)
            
            # 只有非常小的块才考虑合并（<50字符）
            if chunk_len < 50:
                buffer.append(chunk)
                
                # 检查缓冲区是否应该输出
                buffer_total_len = sum(len(c.text) for c in buffer)
                
                # 如果是最后一个块，或者缓冲区已经足够大，输出
                if i == len(chunks) - 1 or buffer_total_len >= self.merge_threshold:
                    merged_chunk = self._merge_buffer(buffer)
                    merged.append(merged_chunk)
                    buffer = []
            else:
                # 当前块足够大（>=50字符）
                if buffer:
                    buffer_total_len = sum(len(c.text) for c in buffer)
                    
                    # 只有在缓冲区很小且合并后不会太大时才合并
                    if buffer_total_len < 50 and (buffer_total_len + chunk_len) < max_merge_size:
                        buffer.append(chunk)
                        merged_chunk = self._merge_buffer(buffer)
                        merged.append(merged_chunk)
                        buffer = []
                    else:
                        # 缓冲区单独输出
                        if buffer_total_len >= self.min_chunk_size:
                            merged_chunk = self._merge_buffer(buffer)
                            merged.append(merged_chunk)
                        buffer = []
                        
                        # 当前块单独保留
                        merged.append(chunk)
                else:
                    # 没有缓冲区，直接添加
                    merged.append(chunk)
        
        # 处理剩余的缓冲区
        if buffer:
            buffer_total_len = sum(len(c.text) for c in buffer)
            if buffer_total_len >= self.min_chunk_size:
                merged_chunk = self._merge_buffer(buffer)
                merged.append(merged_chunk)
        
        return merged
    
    def _merge_buffer(self, buffer: List[DocumentChunk]) -> DocumentChunk:
        """
        合并缓冲区中的块
        
        Args:
            buffer: 待合并的块列表
        
        Returns:
            合并后的块
        """
        if len(buffer) == 1:
            return buffer[0]
        
        # 合并文本（用换行分隔）
        merged_text = "\n".join([c.text for c in buffer])
        
        # 使用第一个块的元数据
        merged_chunk = DocumentChunk(
            chunk_id=buffer[0].chunk_id,
            text=merged_text,
            start_pos=buffer[0].start_pos,
            end_pos=buffer[-1].end_pos,
            page=buffer[0].page,
            section=buffer[0].section,
            context_before=None,
            context_after=None
        )
        
        return merged_chunk
    
    def _add_context_summary(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        为所有块添加上下文摘要
        
        Args:
            chunks: 块列表
        
        Returns:
            添加了上下文的块列表
        """
        for i, chunk in enumerate(chunks):
            # 前文摘要
            if i > 0:
                prev_chunks = chunks[max(0, i-2):i]
                context_before = " | ".join([
                    c.text[:50] + "..." if len(c.text) > 50 else c.text
                    for c in prev_chunks
                ])
                chunk.context_before = context_before
            
            # 后文摘要
            if i < len(chunks) - 1:
                next_chunks = chunks[i+1:min(len(chunks), i+3)]
                context_after = " | ".join([
                    c.text[:50] + "..." if len(c.text) > 50 else c.text
                    for c in next_chunks
                ])
                chunk.context_after = context_after
        
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
    
    def get_chunk_statistics(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """
        获取分块统计信息
        
        Args:
            chunks: 块列表
        
        Returns:
            统计信息
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "total_chars": 0,
                "avg_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0
            }
        
        chunk_sizes = [len(c.text) for c in chunks]
        
        return {
            "total_chunks": len(chunks),
            "total_chars": sum(chunk_sizes),
            "avg_chunk_size": sum(chunk_sizes) // len(chunks),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "size_distribution": {
                "tiny (<50)": sum(1 for s in chunk_sizes if s < 50),
                "small (50-100)": sum(1 for s in chunk_sizes if 50 <= s < 100),
                "medium (100-500)": sum(1 for s in chunk_sizes if 100 <= s < 500),
                "large (500-1000)": sum(1 for s in chunk_sizes if 500 <= s < 1000),
                "huge (>1000)": sum(1 for s in chunk_sizes if s >= 1000)
            }
        }
    
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
    
    def validate_chunks(self, chunks: List[DocumentChunk]) -> List[str]:
        """
        验证分块质量
        
        Args:
            chunks: 块列表
        
        Returns:
            警告信息列表
        """
        warnings = []
        
        for i, chunk in enumerate(chunks):
            chunk_len = len(chunk.text)
            
            # 检查是否有过小的块
            if chunk_len < self.min_chunk_size:
                warnings.append(f"块 {chunk.chunk_id} 太小 ({chunk_len} 字符): {chunk.text[:30]}...")
            
            # 检查是否有过大的块
            if chunk_len > self.chunk_size * 2:
                warnings.append(f"块 {chunk.chunk_id} 太大 ({chunk_len} 字符)，可能需要进一步切分")
            
            # 检查是否只有标点符号
            if all(c in '()（）[]【】{}「」『』<>《》、，。；：！？\n\t ' for c in chunk.text):
                warnings.append(f"块 {chunk.chunk_id} 只包含标点符号: {chunk.text}")
        
        return warnings

