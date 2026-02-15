"""
RAG 检索引擎 - 标准知识库检索
"""
import json
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

from ..models.document import Standard, Rule


class RAGEngine:
    """
    RAG 检索引擎
    
    功能：
    1. 加载标准知识库
    2. 向量化标准规则
    3. 根据文本检索相关规则
    """
    
    def __init__(self, standards_dir: str = "standards/protocols"):
        self.standards_dir = Path(standards_dir)
        self.standards: Dict[str, Standard] = {}
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.rule_vectors = None
        self.rule_index = []  # 规则索引
        
        # 加载标准
        self._load_standards()
        
        # 构建向量索引
        self._build_vector_index()
    
    def _load_standards(self):
        """加载所有标准文件"""
        if not self.standards_dir.exists():
            logger.warning(f"标准目录不存在: {self.standards_dir}")
            return
        
        for file_path in self.standards_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    standard = Standard(**data)
                    self.standards[standard.protocol_id] = standard
                    logger.info(f"加载标准: {standard.name}")
            except Exception as e:
                logger.error(f"加载标准失败 {file_path}: {e}")
    
    def _build_vector_index(self):
        """
        构建向量索引
        
        将所有规则向量化，用于快速检索
        """
        if not self.standards:
            logger.warning("没有加载任何标准，跳过向量索引构建")
            return
        
        # 收集所有规则
        all_rules = []
        for standard in self.standards.values():
            for category in standard.categories:
                for rule in category.rules:
                    all_rules.append({
                        "protocol_id": standard.protocol_id,
                        "category": category.category,
                        "rule": rule
                    })
        
        if not all_rules:
            return
        
        # 构建文本语料（用于向量化）
        corpus = []
        for item in all_rules:
            rule = item["rule"]
            # 组合规则的多个字段
            text = f"{rule.description} {' '.join(rule.keywords)} {' '.join(rule.positive_examples[:2])}"
            corpus.append(text)
            self.rule_index.append(item)
        
        # 向量化
        self.rule_vectors = self.vectorizer.fit_transform(corpus)
        
        logger.info(f"向量索引构建完成: {len(self.rule_index)} 条规则")
    
    def retrieve_relevant_rules(
        self,
        text: str,
        protocol_id: Optional[str] = None,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        检索相关规则
        
        Args:
            text: 待检索文本
            protocol_id: 指定协议ID（如果为空则检索所有）
            top_k: 返回前 k 个最相关的规则
        
        Returns:
            相关规则列表
        """
        if self.rule_vectors is None or not self.rule_index:
            logger.warning("❌ 向量索引未构建，返回空结果")
            return []
        
        logger.debug(f"🔍 RAG 检索: 文本='{text[:50]}...', 协议={protocol_id}")
        
        # 如果指定了协议，先过滤出该协议的规则索引
        if protocol_id:
            protocol_indices = [
                i for i, item in enumerate(self.rule_index)
                if item["protocol_id"] == protocol_id
            ]
            
            if not protocol_indices:
                logger.warning(f"❌ 协议 {protocol_id} 没有任何规则")
                return []
            
            logger.debug(f"   协议 {protocol_id} 共有 {len(protocol_indices)} 条规则")
            
            # 只对该协议的规则计算相似度
            protocol_vectors = self.rule_vectors[protocol_indices]
            query_vector = self.vectorizer.transform([text])
            similarities = cosine_similarity(query_vector, protocol_vectors)[0]
            
            # 获取 top-k
            if len(similarities) < top_k:
                top_k = len(similarities)
            
            top_local_indices = np.argsort(similarities)[-top_k:][::-1]
            top_indices = [protocol_indices[i] for i in top_local_indices]
            top_similarities = similarities[top_local_indices]
        else:
            # 检索所有规则
            query_vector = self.vectorizer.transform([text])
            similarities = cosine_similarity(query_vector, self.rule_vectors)[0]
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            top_similarities = similarities[top_indices]
        
        # 构造结果（降低相似度阈值到 0.01，几乎不过滤）
        results = []
        for idx, similarity in zip(top_indices, top_similarities):
            item = self.rule_index[idx]
            rule = item["rule"]
            
            results.append({
                "rule_id": rule.rule_id,
                "category": item["category"],
                "description": rule.description,
                "check_type": rule.check_type,
                "keywords": rule.keywords,
                "positive_examples": rule.positive_examples,
                "negative_examples": rule.negative_examples,
                "severity": rule.severity,
                "similarity": float(similarity)
            })
            
            logger.debug(f"   ✅ 规则 {rule.rule_id}: {rule.description[:30]}... (相似度: {similarity:.3f})")
        
        if not results:
            logger.warning(f"❌ 没有检索到任何规则！")
        else:
            logger.info(f"✅ 检索到 {len(results)} 条相关规则")
        
        return results
    
    def get_all_rules_by_protocol(self, protocol_id: str) -> List[Dict[str, Any]]:
        """
        获取指定协议的所有规则
        
        Args:
            protocol_id: 协议ID
        
        Returns:
            规则列表
        """
        if protocol_id not in self.standards:
            logger.warning(f"协议不存在: {protocol_id}")
            return []
        
        standard = self.standards[protocol_id]
        rules = []
        
        for category in standard.categories:
            for rule in category.rules:
                rules.append({
                    "rule_id": rule.rule_id,
                    "category": category.category,
                    "description": rule.description,
                    "check_type": rule.check_type,
                    "keywords": rule.keywords,
                    "positive_examples": rule.positive_examples,
                    "negative_examples": rule.negative_examples,
                    "severity": rule.severity
                })
        
        return rules
    
    def list_available_protocols(self) -> List[Dict[str, str]]:
        """
        列出所有可用的协议
        
        Returns:
            协议列表
        """
        return [
            {
                "protocol_id": std.protocol_id,
                "name": std.name,
                "version": std.version,
                "description": std.description or ""
            }
            for std in self.standards.values()
        ]
    
    def save_index(self, file_path: str = "standards/embeddings/index.pkl"):
        """保存向量索引（可选，用于加速启动）"""
        try:
            with open(file_path, 'wb') as f:
                pickle.dump({
                    "vectorizer": self.vectorizer,
                    "rule_vectors": self.rule_vectors,
                    "rule_index": self.rule_index
                }, f)
            logger.info(f"向量索引已保存: {file_path}")
        except Exception as e:
            logger.error(f"保存向量索引失败: {e}")
    
    def load_index(self, file_path: str = "standards/embeddings/index.pkl"):
        """加载向量索引"""
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                self.vectorizer = data["vectorizer"]
                self.rule_vectors = data["rule_vectors"]
                self.rule_index = data["rule_index"]
            logger.info(f"向量索引已加载: {file_path}")
        except Exception as e:
            logger.error(f"加载向量索引失败: {e}")

