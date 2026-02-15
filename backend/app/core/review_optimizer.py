"""
智能审核优化器 - LLM调用优化
"""
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
import hashlib
import re

from ..models.document import DocumentChunk


class SmartReviewOptimizer:
    """
    智能审核优化器
    
    功能：
    1. 智能跳过无需审核的块（减少50%的LLM调用）
    2. 相似块去重（避免重复审核）
    3. 批量优化（提高并发效率）
    4. 缓存机制（避免重复计算）
    
    目标：速度提升 2-3 倍，成本降低 50%
    """
    
    def __init__(self):
        # 缓存：文本哈希 -> 审核结果
        self._review_cache = {}
        
        # 统计信息
        self.stats = {
            "total_chunks": 0,
            "skipped_chunks": 0,
            "cached_chunks": 0,
            "reviewed_chunks": 0,
            "skip_reasons": {}
        }
    
    def should_skip_chunk(
        self,
        chunk: DocumentChunk,
        protocol_id: str,
        rag_engine
    ) -> Tuple[bool, str]:
        """
        判断是否应该跳过这个块（不调用LLM）
        
        Args:
            chunk: 文档块
            protocol_id: 协议ID
            rag_engine: RAG检索引擎
        
        Returns:
            (是否跳过, 跳过原因)
        """
        text = chunk.text.strip()
        
        # 规则1：空文本或太短（<15字符）
        if not text or len(text) < 15:
            return True, "文本太短"
        
        # 规则2：只有标点符号
        if all(c in '()（）[]【】{}「」『』<>《》、，。；：！？\n\t ' for c in text):
            return True, "只有标点符号"
        
        # 规则3：只有数字
        if text.replace(' ', '').replace('\t', '').replace('\n', '').isdigit():
            return True, "只有数字"
        
        # 规则4：只有单个字符重复（如：====、----）
        if len(set(text.replace(' ', '').replace('\n', ''))) <= 2:
            return True, "重复字符"
        
        # 规则5：明显的页眉页脚（如：第X页、共X页）
        if self._is_header_footer(text):
            return True, "页眉页脚"
        
        # 规则6：没有匹配任何规则（RAG检索为空）
        relevant_rules = rag_engine.retrieve_relevant_rules(
            text=text,
            protocol_id=protocol_id,
            top_k=1,
            min_similarity=0.3  # 使用较低阈值，避免漏掉
        )
        
        if not relevant_rules:
            return True, "无匹配规则"
        
        # 规则7：纯表格标题行（如：序号、名称、备注）
        if self._is_table_header(text):
            return True, "表格标题"
        
        # 规则8：纯引用标记（如：[1] [2] 参考文献）
        if self._is_reference_marker(text):
            return True, "引用标记"
        
        return False, ""
    
    def _is_header_footer(self, text: str) -> bool:
        """判断是否是页眉页脚"""
        patterns = [
            r'^第\s*\d+\s*页',
            r'共\s*\d+\s*页',
            r'^\d+\s*/\s*\d+$',
            r'^页码[:：]\s*\d+',
        ]
        
        for pattern in patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def _is_table_header(self, text: str) -> bool:
        """判断是否是表格标题行"""
        # 常见的表格标题关键词
        table_keywords = [
            '序号', '编号', '名称', '类型', '说明', '备注', 
            '日期', '时间', '状态', '结果', '数量', '单位'
        ]
        
        # 如果文本很短且包含多个表格关键词，可能是表头
        if len(text) < 30:
            keyword_count = sum(1 for kw in table_keywords if kw in text)
            if keyword_count >= 2:
                return True
        
        return False
    
    def _is_reference_marker(self, text: str) -> bool:
        """判断是否是引用标记"""
        patterns = [
            r'^\[\d+\]$',
            r'^参考文献$',
            r'^引用$',
            r'^注释[:：]',
        ]
        
        for pattern in patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def deduplicate_chunks(
        self,
        chunks: List[DocumentChunk]
    ) -> Tuple[List[DocumentChunk], List[List[int]]]:
        """
        去重相似的块（避免重复审核）
        
        策略：
        1. 计算每个块的语义哈希
        2. 相同哈希的块只审核一次
        3. 返回去重后的块 + 重复块的索引映射
        
        Args:
            chunks: 原始块列表
        
        Returns:
            (去重后的块列表, 重复块索引映射)
            例如：duplicate_map[0] = [0, 5, 10] 表示块0、5、10内容相同
        """
        hash_to_chunks = {}
        duplicate_map = []
        unique_chunks = []
        
        for i, chunk in enumerate(chunks):
            # 计算语义哈希（忽略空格、换行等）
            text_normalized = chunk.text.replace(' ', '').replace('\n', '').replace('\t', '')
            text_hash = hashlib.md5(text_normalized.encode()).hexdigest()
            
            if text_hash not in hash_to_chunks:
                # 新的唯一块
                hash_to_chunks[text_hash] = len(unique_chunks)
                unique_chunks.append(chunk)
                duplicate_map.append([i])
            else:
                # 重复块
                unique_idx = hash_to_chunks[text_hash]
                duplicate_map[unique_idx].append(i)
        
        # 统计去重效果
        total_duplicates = sum(len(indices) - 1 for indices in duplicate_map)
        if total_duplicates > 0:
            logger.info(
                f"🔄 去重: {len(chunks)} 个块 -> {len(unique_chunks)} 个唯一块 "
                f"(去除 {total_duplicates} 个重复)"
            )
        
        return unique_chunks, duplicate_map
    
    def get_cached_result(self, chunk: DocumentChunk) -> Optional[List]:
        """
        从缓存获取审核结果
        
        Args:
            chunk: 文档块
        
        Returns:
            缓存的审核结果，如果没有则返回 None
        """
        # 计算缓存键
        cache_key = self._get_cache_key(chunk.text)
        
        if cache_key in self._review_cache:
            logger.debug(f"✅ 缓存命中: {chunk.chunk_id}")
            self.stats["cached_chunks"] += 1
            return self._review_cache[cache_key]
        
        return None
    
    def cache_result(self, chunk: DocumentChunk, result: List):
        """
        缓存审核结果
        
        Args:
            chunk: 文档块
            result: 审核结果
        """
        cache_key = self._get_cache_key(chunk.text)
        self._review_cache[cache_key] = result
    
    def _get_cache_key(self, text: str) -> str:
        """生成缓存键"""
        # 标准化文本（忽略空格、换行）
        normalized = text.replace(' ', '').replace('\n', '').replace('\t', '')
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def optimize_batch_size(
        self,
        total_chunks: int,
        avg_chunk_size: int
    ) -> int:
        """
        动态优化批次大小
        
        策略：
        - 小文档（<20块）：批次=5
        - 中文档（20-50块）：批次=8
        - 大文档（>50块）：批次=10
        - 考虑块大小：块越大，批次越小
        
        Args:
            total_chunks: 总块数
            avg_chunk_size: 平均块大小（字符数）
        
        Returns:
            优化后的批次大小
        """
        # 基础批次大小
        if total_chunks < 20:
            base_batch_size = 5
        elif total_chunks < 50:
            base_batch_size = 8
        else:
            base_batch_size = 10
        
        # 根据块大小调整
        if avg_chunk_size > 800:
            # 块很大，减小批次
            batch_size = max(3, base_batch_size - 2)
        elif avg_chunk_size > 500:
            batch_size = max(4, base_batch_size - 1)
        else:
            batch_size = base_batch_size
        
        logger.info(
            f"📦 批次优化: 总块数={total_chunks}, 平均大小={avg_chunk_size}, "
            f"批次大小={batch_size}"
        )
        
        return batch_size
    
    def filter_chunks_for_review(
        self,
        chunks: List[DocumentChunk],
        protocol_id: str,
        rag_engine
    ) -> Tuple[List[DocumentChunk], Dict[str, Any]]:
        """
        过滤需要审核的块（综合优化）
        
        流程：
        1. 智能跳过（无需审核的块）
        2. 去重（相同内容的块）
        3. 缓存检查（已审核过的块）
        
        Args:
            chunks: 原始块列表
            protocol_id: 协议ID
            rag_engine: RAG引擎
        
        Returns:
            (需要审核的块列表, 优化信息)
        """
        self.stats["total_chunks"] = len(chunks)
        
        # 第一步：智能跳过
        chunks_to_review = []
        skipped_info = []
        
        for chunk in chunks:
            should_skip, reason = self.should_skip_chunk(chunk, protocol_id, rag_engine)
            
            if should_skip:
                self.stats["skipped_chunks"] += 1
                self.stats["skip_reasons"][reason] = self.stats["skip_reasons"].get(reason, 0) + 1
                skipped_info.append({
                    "chunk_id": chunk.chunk_id,
                    "reason": reason,
                    "text_preview": chunk.text[:50]
                })
                logger.debug(f"⏭️  跳过块 {chunk.chunk_id}: {reason}")
            else:
                chunks_to_review.append(chunk)
        
        logger.info(
            f"🎯 智能跳过: {len(chunks)} 个块 -> {len(chunks_to_review)} 个需审核 "
            f"(跳过 {self.stats['skipped_chunks']} 个)"
        )
        
        # 第二步：去重
        unique_chunks, duplicate_map = self.deduplicate_chunks(chunks_to_review)
        
        # 第三步：缓存检查（可选，如果启用缓存）
        chunks_need_llm = []
        cached_results = {}
        
        for chunk in unique_chunks:
            cached = self.get_cached_result(chunk)
            if cached is not None:
                cached_results[chunk.chunk_id] = cached
            else:
                chunks_need_llm.append(chunk)
        
        if cached_results:
            logger.info(f"💾 缓存命中: {len(cached_results)} 个块")
        
        # 统计信息
        optimization_info = {
            "original_count": len(chunks),
            "skipped_count": self.stats["skipped_chunks"],
            "skip_reasons": self.stats["skip_reasons"],
            "deduplicated_count": len(chunks_to_review) - len(unique_chunks),
            "cached_count": len(cached_results),
            "final_review_count": len(chunks_need_llm),
            "optimization_rate": (1 - len(chunks_need_llm) / len(chunks)) * 100 if len(chunks) > 0 else 0,
            "skipped_chunks": skipped_info,
            "duplicate_map": duplicate_map,
            "cached_results": cached_results
        }
        
        logger.info(
            f"✨ 优化完成: {len(chunks)} -> {len(chunks_need_llm)} 个块需要LLM审核 "
            f"(优化率: {optimization_info['optimization_rate']:.1f}%)"
        )
        
        return chunks_need_llm, optimization_info
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取优化统计信息"""
        total = self.stats["total_chunks"]
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            "skip_rate": self.stats["skipped_chunks"] / total * 100,
            "cache_hit_rate": self.stats["cached_chunks"] / total * 100,
            "review_rate": self.stats["reviewed_chunks"] / total * 100
        }
    
    def clear_cache(self):
        """清空缓存"""
        self._review_cache.clear()
        logger.info("🗑️  缓存已清空")
    
    def get_cache_size(self) -> int:
        """获取缓存大小"""
        return len(self._review_cache)

