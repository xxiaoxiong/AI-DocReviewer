"""
数据模型定义
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class CheckType(str, Enum):
    """检查类型"""
    SEMANTIC = "semantic"  # 语义检查
    FORMAT = "format"      # 格式检查
    STRUCTURE = "structure"  # 结构检查


class Severity(str, Enum):
    """严重程度"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Rule(BaseModel):
    """规则定义"""
    rule_id: str
    description: str
    check_type: CheckType
    keywords: List[str] = []
    positive_examples: List[str] = []
    negative_examples: List[str] = []
    severity: Severity = Severity.MEDIUM


class Category(BaseModel):
    """规则分类"""
    category: str
    rules: List[Rule]


class Standard(BaseModel):
    """标准/协议定义"""
    protocol_id: str
    name: str
    version: str
    description: Optional[str] = None
    categories: List[Category]


class DocumentChunk(BaseModel):
    """文档块"""
    chunk_id: str
    text: str
    start_pos: int
    end_pos: int
    page: Optional[int] = None
    section: Optional[str] = None  # 章节标题
    context_before: Optional[str] = None  # 前文摘要
    context_after: Optional[str] = None   # 后文摘要


class Issue(BaseModel):
    """检测到的问题"""
    issue_id: str
    position: str  # 位置描述
    page: Optional[int] = None
    rule_id: str
    category: str
    original_text: str
    issue_description: str
    suggestion: str
    confidence: float = Field(ge=0.0, le=1.0)
    severity: Severity


class ReviewResult(BaseModel):
    """审核结果"""
    document_id: str
    protocol_id: str
    total_issues: int
    issues: List[Issue]
    summary: Dict[str, Any]
    created_at: str

