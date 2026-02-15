"""
审核日志记录器 - 记录所有 LLM 调用和结果
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger


class ReviewLogger:
    """审核日志记录器"""
    
    def __init__(self, log_dir: str = "logs/reviews"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_session = None
        self.session_logs = []
    
    def start_session(self, document_name: str, protocol_id: str) -> str:
        """
        开始新的审核会话
        
        Args:
            document_name: 文档名称
            protocol_id: 协议ID
        
        Returns:
            会话ID
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"{timestamp}_{document_name.replace('.', '_')}"
        
        self.current_session = {
            "session_id": session_id,
            "document_name": document_name,
            "protocol_id": protocol_id,
            "start_time": datetime.now().isoformat(),
            "chunks": [],
            "total_llm_calls": 0,
            "total_issues_found": 0
        }
        
        self.session_logs = []
        logger.info(f"📝 开始审核会话: {session_id}")
        return session_id
    
    def log_chunk_review(
        self,
        chunk_id: str,
        chunk_text: str,
        relevant_rules: List[Dict[str, Any]],
        llm_prompt: str,
        llm_response: Dict[str, Any],
        issues_found: int,
        error: Optional[str] = None
    ):
        """
        记录单个块的审核过程
        
        Args:
            chunk_id: 块ID
            chunk_text: 块文本
            relevant_rules: 相关规则
            llm_prompt: 发送给 LLM 的 prompt
            llm_response: LLM 返回的响应
            issues_found: 发现的问题数
            error: 错误信息（如果有）
        """
        log_entry = {
            "chunk_id": chunk_id,
            "timestamp": datetime.now().isoformat(),
            "chunk_text": chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text,
            "chunk_length": len(chunk_text),
            "relevant_rules_count": len(relevant_rules),
            "relevant_rules": [
                {
                    "rule_id": r.get("rule_id"),
                    "category": r.get("category"),
                    "description": r.get("description")
                }
                for r in relevant_rules
            ],
            "llm_prompt_length": len(llm_prompt),
            "llm_prompt": llm_prompt,  # 完整 prompt
            "llm_response": llm_response,  # 完整响应
            "issues_found": issues_found,
            "error": error,
            "success": error is None
        }
        
        self.session_logs.append(log_entry)
        
        if self.current_session:
            self.current_session["chunks"].append(log_entry)
            self.current_session["total_llm_calls"] += 1
            self.current_session["total_issues_found"] += issues_found
        
        # 实时保存（防止崩溃丢失数据）
        self._save_current_chunk(log_entry)
        
        logger.info(f"📊 块 {chunk_id}: 调用 LLM ✅, 发现 {issues_found} 个问题")
    
    def end_session(self, final_result: Dict[str, Any]):
        """
        结束审核会话并保存完整日志
        
        Args:
            final_result: 最终审核结果
        """
        if not self.current_session:
            logger.warning("没有活动的审核会话")
            return
        
        self.current_session["end_time"] = datetime.now().isoformat()
        self.current_session["final_result"] = final_result
        
        # 计算统计信息
        self.current_session["statistics"] = self._calculate_statistics()
        
        # 保存完整日志
        self._save_session_log()
        
        logger.info(f"✅ 审核会话结束: {self.current_session['session_id']}")
        logger.info(f"   - 总调用次数: {self.current_session['total_llm_calls']}")
        logger.info(f"   - 发现问题数: {self.current_session['total_issues_found']}")
        
        self.current_session = None
        self.session_logs = []
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """计算统计信息"""
        if not self.session_logs:
            return {}
        
        total_chunks = len(self.session_logs)
        successful_calls = sum(1 for log in self.session_logs if log["success"])
        failed_calls = total_chunks - successful_calls
        
        total_issues = sum(log["issues_found"] for log in self.session_logs)
        chunks_with_issues = sum(1 for log in self.session_logs if log["issues_found"] > 0)
        
        return {
            "total_chunks": total_chunks,
            "successful_llm_calls": successful_calls,
            "failed_llm_calls": failed_calls,
            "success_rate": f"{successful_calls / total_chunks * 100:.1f}%",
            "total_issues_found": total_issues,
            "chunks_with_issues": chunks_with_issues,
            "average_issues_per_chunk": f"{total_issues / total_chunks:.2f}",
            "issue_detection_rate": f"{chunks_with_issues / total_chunks * 100:.1f}%"
        }
    
    def _save_current_chunk(self, log_entry: Dict[str, Any]):
        """实时保存当前块的日志"""
        if not self.current_session:
            return
        
        session_id = self.current_session["session_id"]
        chunk_log_file = self.log_dir / f"{session_id}_chunks.jsonl"
        
        try:
            with open(chunk_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"保存块日志失败: {e}")
    
    def _save_session_log(self):
        """保存完整的会话日志"""
        if not self.current_session:
            return
        
        session_id = self.current_session["session_id"]
        
        # 保存完整日志（JSON 格式，便于查看）
        full_log_file = self.log_dir / f"{session_id}_full.json"
        try:
            with open(full_log_file, 'w', encoding='utf-8') as f:
                json.dump(self.current_session, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 完整日志已保存: {full_log_file}")
        except Exception as e:
            logger.error(f"保存完整日志失败: {e}")
        
        # 保存摘要（便于快速查看）
        summary_file = self.log_dir / f"{session_id}_summary.txt"
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(f"审核会话摘要: {session_id}\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"文档名称: {self.current_session['document_name']}\n")
                f.write(f"使用协议: {self.current_session['protocol_id']}\n")
                f.write(f"开始时间: {self.current_session['start_time']}\n")
                f.write(f"结束时间: {self.current_session['end_time']}\n\n")
                
                stats = self.current_session['statistics']
                f.write("统计信息:\n")
                f.write(f"  - 总块数: {stats['total_chunks']}\n")
                f.write(f"  - 成功调用: {stats['successful_llm_calls']}\n")
                f.write(f"  - 失败调用: {stats['failed_llm_calls']}\n")
                f.write(f"  - 成功率: {stats['success_rate']}\n")
                f.write(f"  - 发现问题数: {stats['total_issues_found']}\n")
                f.write(f"  - 有问题的块: {stats['chunks_with_issues']}\n")
                f.write(f"  - 平均每块问题数: {stats['average_issues_per_chunk']}\n")
                f.write(f"  - 问题检出率: {stats['issue_detection_rate']}\n\n")
                
                f.write("=" * 80 + "\n")
                f.write("详细结果\n")
                f.write("=" * 80 + "\n\n")
                
                for i, chunk_log in enumerate(self.session_logs, 1):
                    f.write(f"块 {i}: {chunk_log['chunk_id']}\n")
                    f.write(f"  文本: {chunk_log['chunk_text']}\n")
                    f.write(f"  规则数: {chunk_log['relevant_rules_count']}\n")
                    f.write(f"  发现问题: {chunk_log['issues_found']}\n")
                    if chunk_log['error']:
                        f.write(f"  错误: {chunk_log['error']}\n")
                    f.write("\n")
            
            logger.info(f"📄 摘要已保存: {summary_file}")
        except Exception as e:
            logger.error(f"保存摘要失败: {e}")
    
    def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取最近的审核会话
        
        Args:
            limit: 返回数量
        
        Returns:
            会话列表
        """
        summary_files = sorted(
            self.log_dir.glob("*_summary.txt"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )[:limit]
        
        sessions = []
        for file in summary_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    sessions.append({
                        "file": file.name,
                        "path": str(file),
                        "modified": datetime.fromtimestamp(file.stat().st_mtime).isoformat()
                    })
            except Exception as e:
                logger.error(f"读取会话文件失败 {file}: {e}")
        
        return sessions


# 全局实例
review_logger = ReviewLogger()

