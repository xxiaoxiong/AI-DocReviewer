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

