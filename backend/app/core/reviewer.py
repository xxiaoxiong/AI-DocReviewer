"""
文档审核器 - 核心审核引擎
"""
import asyncio
from typing import List, Dict, Any, Optional
from loguru import logger
from datetime import datetime
import uuid
from pathlib import Path

from ..models.document import DocumentChunk, Issue, ReviewResult, Severity
from ..services.llm_service import LLMService
from .document_parser import DocumentParser
from .chunker import SmartChunker
from .rag_engine import RAGEngine
from .review_logger import review_logger


class DocumentReviewer:
    """
    文档审核器
    
    核心流程：
    1. 解析文档
    2. 智能分块
    3. RAG 检索相关规则
    4. LLM 审核
    5. 结果聚合
    """
    
    def __init__(
        self,
        rag_engine: RAGEngine,
        llm_service: LLMService,
        use_cross_chunk_check: bool = True
    ):
        self.parser = DocumentParser()
        self.chunker = SmartChunker()
        self.rag = rag_engine
        self.llm = llm_service
        self.use_cross_chunk_check = use_cross_chunk_check
    
    async def review_document(
        self,
        file_path: str,
        protocol_id: str,
        batch_size: int = 5
    ) -> ReviewResult:
        """
        审核完整文档
        
        Args:
            file_path: 文档路径
            protocol_id: 使用的协议ID
            batch_size: 批量处理大小
        
        Returns:
            审核结果
        """
        document_id = str(uuid.uuid4())
        document_name = Path(file_path).name
        logger.info(f"开始审核文档: {file_path}, 协议: {protocol_id}")
        
        # 开始日志会话
        session_id = review_logger.start_session(document_name, protocol_id)
        
        try:
            # 1. 解析文档
            doc_structure = self.parser.parse_docx(file_path)
            
            # 2. 智能分块
            chunks = self.chunker.chunk_by_paragraphs(doc_structure)
            logger.info(f"文档分块完成: {len(chunks)} 个块")
            
            # 3. 批量审核
            all_issues = []
            total_chunks = len(chunks)
            
            logger.info(f"📋 开始逐块审核，共 {total_chunks} 个块")
            logger.info(f"   批次大小: {batch_size}")
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                batch_num = i // batch_size + 1
                total_batches = (total_chunks + batch_size - 1) // batch_size
                
                logger.info("=" * 60)
                logger.info(f"📦 批次 {batch_num}/{total_batches}: 审核块 {i+1}-{min(i+len(batch), total_chunks)}/{total_chunks}")
                logger.info("=" * 60)
                
                # 并行审核当前批次
                tasks = [
                    self._review_chunk(chunk, protocol_id)
                    for chunk in batch
                ]
                
                # 使用 return_exceptions=False 确保异常会被抛出
                try:
                    batch_results = await asyncio.gather(*tasks, return_exceptions=False)
                    
                    # 收集结果
                    batch_issues = 0
                    for issues in batch_results:
                        all_issues.extend(issues)
                        batch_issues += len(issues)
                    
                    logger.info(f"✅ 批次 {batch_num} 完成，发现 {batch_issues} 个问题")
                    
                except Exception as e:
                    logger.error(f"❌ 批次 {batch_num} 审核失败: {e}")
                    # 记录失败并继续抛出
                    raise Exception(f"审核失败: {str(e)}。请检查：1) API Key 是否正确 2) 网络连接 3) 查看终端详细日志")
            
            # 4. 跨段落二次检查（可选）
            if self.use_cross_chunk_check:
                cross_issues = await self._cross_chunk_check(chunks, protocol_id, all_issues)
                all_issues.extend(cross_issues)
            
            # 5. 去重和排序
            unique_issues = self._deduplicate_issues(all_issues)
            unique_issues.sort(key=lambda x: (x.page or 0, x.position))
            
            # 6. 生成摘要
            summary = self._generate_summary(unique_issues)
            
            result = ReviewResult(
                document_id=document_id,
                protocol_id=protocol_id,
                total_issues=len(unique_issues),
                issues=unique_issues,
                summary=summary,
                created_at=datetime.now().isoformat()
            )
            
            logger.info(f"审核完成: 发现 {len(unique_issues)} 个问题")
            
            # 结束日志会话
            review_logger.end_session(result.dict())
            
            return result
        
        except Exception as e:
            logger.error(f"审核失败: {e}")
            # 即使失败也保存日志
            review_logger.end_session({"error": str(e)})
            raise
    
    async def _review_chunk(
        self,
        chunk: DocumentChunk,
        protocol_id: str
    ) -> List[Issue]:
        """
        审核单个文本块
        
        Args:
            chunk: 文档块
            protocol_id: 协议ID
        
        Returns:
            问题列表
        """
        import time
        start_time = time.time()
        
        error_msg = None
        llm_prompt = ""
        llm_response = {}
        relevant_rules = []
        
        try:
            logger.info(f"🔍 审核块: {chunk.chunk_id}")
            logger.debug(f"   文本: {chunk.text[:100]}...")
            
            # 1. 检索相关规则
            relevant_rules = self.rag.retrieve_relevant_rules(
                text=chunk.text,
                protocol_id=protocol_id,
                top_k=3
            )
            
            if not relevant_rules:
                logger.debug(f"   ⚠️  没有匹配的规则，跳过")
                # 记录日志：没有匹配规则
                review_logger.log_chunk_review(
                    chunk_id=chunk.chunk_id,
                    chunk_text=chunk.text,
                    relevant_rules=[],
                    llm_prompt="",
                    llm_response={"issues": [], "note": "没有匹配的规则"},
                    issues_found=0
                )
                return []
            
            logger.info(f"   📚 匹配到 {len(relevant_rules)} 条规则")
            for rule in relevant_rules:
                logger.debug(f"      - {rule['rule_id']}: {rule['description'][:50]}...")
            
            # 2. 构造上下文
            context = None
            if chunk.context_before or chunk.context_after:
                context = f"前文: {chunk.context_before or '无'}\n后文: {chunk.context_after or '无'}"
            
            # 3. 调用 LLM 审核
            logger.info(f"   🤖 调用 LLM 进行审核...")
            
            # 获取 prompt（用于日志记录）
            llm_prompt = self.llm._build_review_prompt(chunk.text, relevant_rules, context)
            
            result = await self.llm.review_chunk(
                text=chunk.text,
                relevant_rules=relevant_rules,
                context=context
            )
            llm_response = result
            
            elapsed = time.time() - start_time
            
            # 4. 解析结果
            issues = []
            for item in result.get("issues", []):
                issue = Issue(
                    issue_id=str(uuid.uuid4()),
                    position=item.get("position", ""),
                    page=chunk.page,
                    rule_id=item.get("rule_id", ""),
                    category=item.get("category", ""),
                    original_text=item.get("original_text", ""),
                    issue_description=item.get("issue_description", ""),
                    suggestion=item.get("suggestion", ""),
                    confidence=item.get("confidence", 0.5),
                    severity=Severity(item.get("severity", "medium"))
                )
                
                # 只保留高置信度的问题
                if issue.confidence >= 0.7:
                    issues.append(issue)
            
            if issues:
                logger.info(f"   ⚠️  发现 {len(issues)} 个问题 (耗时: {elapsed:.2f}s)")
                for issue in issues:
                    logger.debug(f"      - [{issue.severity.value}] {issue.category}: {issue.issue_description[:50]}...")
            else:
                logger.info(f"   ✅ 未发现问题 (耗时: {elapsed:.2f}s)")
            
            # 记录成功的审核日志
            review_logger.log_chunk_review(
                chunk_id=chunk.chunk_id,
                chunk_text=chunk.text,
                relevant_rules=relevant_rules,
                llm_prompt=llm_prompt,
                llm_response=llm_response,
                issues_found=len(issues)
            )
            
            return issues
        
        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = str(e)
            logger.error(f"   ❌ 审核失败 (耗时: {elapsed:.2f}s): {e}")
            import traceback
            error_detail = traceback.format_exc()
            logger.error(f"   详细错误:\n{error_detail}")
            
            # 记录失败的审核日志
            review_logger.log_chunk_review(
                chunk_id=chunk.chunk_id,
                chunk_text=chunk.text,
                relevant_rules=relevant_rules,
                llm_prompt=llm_prompt,
                llm_response=llm_response,
                issues_found=0,
                error=error_detail
            )
            
            # 重新抛出异常，让上层知道出错了
            raise
    
    async def _cross_chunk_check(
        self,
        chunks: List[DocumentChunk],
        protocol_id: str,
        existing_issues: List[Issue]
    ) -> List[Issue]:
        """
        跨段落检查
        
        解决语义断裂问题的关键！
        
        策略：
        1. 检测可能的跨段落问题（如：引用、逻辑连贯性）
        2. 扩大上下文窗口重新审核
        
        Args:
            chunks: 所有文档块
            protocol_id: 协议ID
            existing_issues: 已发现的问题
        
        Returns:
            新发现的问题
        """
        logger.info("开始跨段落检查...")
        
        # 识别需要跨段落检查的规则类型
        cross_chunk_rules = self.rag.retrieve_relevant_rules(
            text="逻辑连贯性 引用完整性 前后呼应",
            protocol_id=protocol_id,
            top_k=5
        )
        
        if not cross_chunk_rules:
            return []
        
        new_issues = []
        
        # 检查相邻段落
        for i in range(len(chunks) - 1):
            current = chunks[i]
            next_chunk = chunks[i + 1]
            
            # 合并相邻段落
            combined_text = f"{current.text}\n{next_chunk.text}"
            
            # 检查是否有跨段落问题
            try:
                result = await self.llm.review_chunk(
                    text=combined_text,
                    relevant_rules=cross_chunk_rules,
                    context=f"这是相邻的两个段落，请检查它们之间的逻辑连贯性"
                )
                
                for item in result.get("issues", []):
                    # 避免重复
                    if not self._is_duplicate(item, existing_issues):
                        issue = Issue(
                            issue_id=str(uuid.uuid4()),
                            position=f"段落 {i} 和 {i+1} 之间",
                            page=current.page,
                            rule_id=item.get("rule_id", ""),
                            category=item.get("category", "跨段落问题"),
                            original_text=item.get("original_text", "")[:100],
                            issue_description=item.get("issue_description", ""),
                            suggestion=item.get("suggestion", ""),
                            confidence=item.get("confidence", 0.5),
                            severity=Severity(item.get("severity", "medium"))
                        )
                        
                        if issue.confidence >= 0.8:  # 更高的阈值
                            new_issues.append(issue)
            
            except Exception as e:
                logger.error(f"跨段落检查失败 ({i}, {i+1}): {e}")
        
        logger.info(f"跨段落检查完成: 发现 {len(new_issues)} 个新问题")
        return new_issues
    
    def _deduplicate_issues(self, issues: List[Issue]) -> List[Issue]:
        """
        去重
        
        Args:
            issues: 问题列表
        
        Returns:
            去重后的问题列表
        """
        seen = set()
        unique = []
        
        for issue in issues:
            # 使用原文和问题描述作为唯一标识
            key = f"{issue.original_text[:50]}_{issue.issue_description[:50]}"
            
            if key not in seen:
                seen.add(key)
                unique.append(issue)
        
        return unique
    
    def _is_duplicate(self, new_item: Dict, existing_issues: List[Issue]) -> bool:
        """检查是否重复"""
        new_text = new_item.get("original_text", "")[:50]
        
        for issue in existing_issues:
            if issue.original_text[:50] == new_text:
                return True
        
        return False
    
    def _generate_summary(self, issues: List[Issue]) -> Dict[str, Any]:
        """
        生成审核摘要
        
        Args:
            issues: 问题列表
        
        Returns:
            摘要信息
        """
        summary = {
            "total": len(issues),
            "by_severity": {
                "high": 0,
                "medium": 0,
                "low": 0
            },
            "by_category": {}
        }
        
        for issue in issues:
            # 按严重程度统计
            summary["by_severity"][issue.severity.value] += 1
            
            # 按类别统计
            category = issue.category
            if category not in summary["by_category"]:
                summary["by_category"][category] = 0
            summary["by_category"][category] += 1
        
        return summary

