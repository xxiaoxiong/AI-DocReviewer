"""
置信度校准器 - 减少误报
"""
from typing import Dict, Any
from loguru import logger

from ..models.document import Issue, Severity


class ConfidenceCalibrator:
    """
    置信度校准器
    
    功能：
    1. 根据规则类型调整置信度
    2. 根据文本特征调整置信度 
    3. 根据历史准确率调整置信度
    4. 动态阈值过滤
    
    目标：减少误报 50%
    """
    
    def __init__(self):
        # 规则类型权重（基于经验）
        self.rule_type_weights = {
            "format": 1.2,      # 格式类规则更可靠（如：标点、空格）
            "semantic": 0.85,   # 语义类规则需要更谨慎（如：逻辑连贯性）
            "structure": 1.0,   # 结构类规则标准权重（如：章节顺序）
            "content": 0.9      # 内容类规则稍微谨慎（如：用词规范）
        }
        
        # 严重度权重（高严重度要求更高置信度）
        self.severity_weights = {
            "high": 1.1,    # 高严重度问题要求更确定
            "medium": 1.0,  # 中等严重度标准
            "low": 0.9      # 低严重度可以宽松一些
        }
        
        # 历史准确率（可以从日志中学习）
        self.rule_accuracy_history = {}
    
    def calibrate_issue(
        self,
        issue: Issue,
        rule_type: str,
        chunk_text: str,
        context: Dict[str, Any] = None
    ) -> Issue:
        """
        校准单个问题的置信度
        
        Args:
            issue: 原始问题
            rule_type: 规则类型
            chunk_text: 原始文本块
            context: 上下文信息
        
        Returns:
            校准后的问题
        """
        original_confidence = issue.confidence
        
        # 1. 规则类型权重
        type_weight = self.rule_type_weights.get(rule_type, 1.0)
        
        # 2. 严重度权重
        severity_weight = self.severity_weights.get(issue.severity.value, 1.0)
        
        # 3. 文本长度权重（太短的原文可能不可靠）
        length_weight = self._calculate_length_weight(issue.original_text)
        
        # 4. 上下文一致性权重
        context_weight = self._calculate_context_weight(issue, chunk_text, context)
        
        # 5. 历史准确率权重
        history_weight = self._get_history_weight(issue.rule_id)
        
        # 综合校准
        calibrated_confidence = (
            original_confidence 
            * type_weight 
            * severity_weight 
            * length_weight 
            * context_weight 
            * history_weight
        )
        
        # 限制在 [0, 1] 范围
        calibrated_confidence = max(0.0, min(1.0, calibrated_confidence))
        
        # 记录校准信息
        if abs(calibrated_confidence - original_confidence) > 0.1:
            logger.debug(
                f"置信度校准: {issue.rule_id} "
                f"{original_confidence:.2f} -> {calibrated_confidence:.2f} "
                f"(类型:{type_weight:.2f}, 严重度:{severity_weight:.2f}, "
                f"长度:{length_weight:.2f}, 上下文:{context_weight:.2f}, "
                f"历史:{history_weight:.2f})"
            )
        
        # 更新置信度
        issue.confidence = calibrated_confidence
        
        return issue
    
    def _calculate_length_weight(self, original_text: str) -> float:
        """
        计算文本长度权重
        
        原则：
        - 太短（<10字符）：置信度降低（可能是误报）
        - 适中（10-50字符）：标准权重
        - 太长（>100字符）：置信度略微降低（可能包含多个问题）
        """
        length = len(original_text.strip())
        
        if length < 10:
            # 太短，可能是误报
            return 0.6
        elif length < 20:
            # 较短，稍微降低
            return 0.8
        elif length <= 50:
            # 适中，标准权重
            return 1.0
        elif length <= 100:
            # 较长，略微降低
            return 0.95
        else:
            # 太长，可能不够精确
            return 0.85
    
    def _calculate_context_weight(
        self,
        issue: Issue,
        chunk_text: str,
        context: Dict[str, Any]
    ) -> float:
        """
        计算上下文一致性权重
        
        检查：
        1. 原文是否真的在文本块中
        2. 问题描述是否与原文匹配
        3. 建议是否合理
        """
        weight = 1.0
        
        # 检查1：原文是否在文本块中
        if issue.original_text not in chunk_text:
            logger.warning(f"原文不在文本块中: {issue.original_text[:30]}...")
            weight *= 0.5  # 大幅降低置信度
        
        # 检查2：原文是否只有标点符号（可能是误报）
        if all(c in '()（）[]【】{}「」『』<>《》、，。；：！？\n\t ' for c in issue.original_text):
            logger.debug(f"原文只有标点符号: {issue.original_text}")
            weight *= 0.3  # 大幅降低
        
        # 检查3：原文是否只有数字（可能是误报）
        if issue.original_text.strip().replace(' ', '').isdigit():
            logger.debug(f"原文只有数字: {issue.original_text}")
            weight *= 0.4
        
        # 检查4：问题描述是否太短（可能不够具体）
        if len(issue.issue_description) < 10:
            logger.debug(f"问题描述太短: {issue.issue_description}")
            weight *= 0.8
        
        # 检查5：建议是否太短（可能不够具体）
        if len(issue.suggestion) < 10:
            logger.debug(f"建议太短: {issue.suggestion}")
            weight *= 0.9
        
        return weight
    
    def _get_history_weight(self, rule_id: str) -> float:
        """
        获取历史准确率权重
        
        如果某条规则历史上误报率高，降低其置信度
        """
        if rule_id not in self.rule_accuracy_history:
            return 1.0  # 没有历史数据，使用标准权重
        
        accuracy = self.rule_accuracy_history[rule_id]
        
        # 准确率越低，权重越低
        if accuracy < 0.5:
            return 0.7
        elif accuracy < 0.7:
            return 0.85
        else:
            return 1.0
    
    def should_filter_issue(
        self,
        issue: Issue,
        min_confidence: float = 0.7
    ) -> bool:
        """
        判断是否应该过滤掉这个问题
        
        Args:
            issue: 问题
            min_confidence: 最小置信度阈值
        
        Returns:
            True 表示应该过滤（不报告）
        """
        # 基础阈值过滤
        if issue.confidence < min_confidence:
            return True
        
        # 高严重度问题：更严格的阈值
        if issue.severity == Severity.HIGH and issue.confidence < 0.85:
            logger.debug(f"高严重度问题置信度不足: {issue.confidence:.2f} < 0.85")
            return True
        
        # 低严重度问题：可以宽松一些
        if issue.severity == Severity.LOW and issue.confidence >= 0.65:
            return False
        
        return False
    
    def batch_calibrate(
        self,
        issues: list[Issue],
        rule_types: Dict[str, str],
        chunk_text: str,
        context: Dict[str, Any] = None
    ) -> list[Issue]:
        """
        批量校准问题列表
        
        Args:
            issues: 问题列表
            rule_types: 规则ID到类型的映射
            chunk_text: 原始文本块
            context: 上下文信息
        
        Returns:
            校准后的问题列表（已过滤低置信度）
        """
        calibrated_issues = []
        filtered_count = 0
        
        for issue in issues:
            rule_type = rule_types.get(issue.rule_id, "semantic")
            
            # 校准
            calibrated_issue = self.calibrate_issue(
                issue, rule_type, chunk_text, context
            )
            
            # 过滤
            if not self.should_filter_issue(calibrated_issue):
                calibrated_issues.append(calibrated_issue)
            else:
                filtered_count += 1
                logger.debug(
                    f"过滤低置信度问题: {issue.rule_id} "
                    f"({calibrated_issue.confidence:.2f}) - {issue.issue_description[:50]}..."
                )
        
        if filtered_count > 0:
            logger.info(f"📊 置信度校准: 过滤了 {filtered_count}/{len(issues)} 个低置信度问题")
        
        return calibrated_issues
    
    def update_history(self, rule_id: str, is_correct: bool):
        """
        更新规则的历史准确率
        
        Args:
            rule_id: 规则ID
            is_correct: 这次检测是否正确
        """
        if rule_id not in self.rule_accuracy_history:
            self.rule_accuracy_history[rule_id] = {
                "correct": 0,
                "total": 0,
                "accuracy": 1.0
            }
        
        history = self.rule_accuracy_history[rule_id]
        history["total"] += 1
        if is_correct:
            history["correct"] += 1
        
        # 更新准确率（使用滑动平均，避免早期数据影响过大）
        history["accuracy"] = history["correct"] / history["total"]
        
        logger.debug(
            f"更新规则历史: {rule_id} "
            f"准确率={history['accuracy']:.2f} ({history['correct']}/{history['total']})"
        )

