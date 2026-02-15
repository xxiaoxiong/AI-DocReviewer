"""
RAG 检索引擎 V2 - 使用语义嵌入模型（BGE）

升级点：
1. TF-IDF → 深度学习语义嵌入
2. 理解语义，支持同义词
3. 中文友好，检索更准确
"""
import json
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from loguru import logger
import pickle
import faiss

from ..models.document import Standard, Rule


class RAGEngineV2:
    """
    RAG 检索引擎 V2 - 语义检索版本
    
    功能：
    1. 加载标准知识库
    2. 使用语义嵌入模型向量化规则（支持千问3/BGE）
    3. 使用 FAISS 进行高效向量检索
    4. 混合检索（语义+关键词）
    
    推荐模型：
    - BAAI/bge-small-zh-v1.5 (BGE) - 轻量级，演示版本推荐
      * 模型小（~100MB），启动快
      * 适合资源受限环境
    
    """
    
    def __init__(
        self, 
        standards_dir: str = "standards/protocols",
        model_name: str = "BAAI/bge-small-zh-v1.5",  # BGE 轻量级模型（演示版本）
        # model_name: str = "Alibaba-NLP/gte-Qwen2-1.5B-instruct",  # 千问3（生产环境）
        use_faiss: bool = True
    ):
        self.standards_dir = Path(standards_dir)
        self.standards: Dict[str, Standard] = {}
        self.model_name = model_name
        self.use_faiss = use_faiss
        
        # 延迟加载模型（避免启动时加载）
        self.model = None
        self.rule_vectors = None
        self.rule_index = []  # 规则索引
        self.faiss_index = None
        
        # 加载标准
        self._load_standards()
        
        # 初始化模型
        self._init_model()
        
        # 构建向量索引
        self._build_vector_index()
    
    def _init_model(self):
        """初始化嵌入模型"""
        try:
            import os
            from sentence_transformers import SentenceTransformer
            
            # 设置国内镜像源（解决网络问题）
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            
            logger.info(f"🤖 加载语义嵌入模型: {self.model_name}")
            
            # 根据模型给出下载提示
            if "qwen" in self.model_name.lower():
                logger.info("   首次运行会自动下载千问3模型（约 3GB），请稍候...")
                logger.info("   💡 千问3支持超长文本（8192 tokens），检索更准确")
            else:
                logger.info("   首次运行会自动下载模型（约 100-400MB），请稍候...")
            
            logger.info("   使用镜像源: https://hf-mirror.com")
            
            self.model = SentenceTransformer(self.model_name)
            
            logger.info(f"✅ 模型加载完成")
            logger.info(f"   模型维度: {self.model.get_sentence_embedding_dimension()}")
            
        except ImportError:
            logger.error("❌ 未安装 sentence-transformers，请运行：")
            logger.error("   pip install sentence-transformers torch")
            raise
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            logger.error("   如果是网络问题，可以手动下载模型到本地")
            raise
    
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
        构建向量索引（使用语义嵌入）
        
        将所有规则向量化，用于快速检索
        """
        if not self.standards:
            logger.warning("没有加载任何标准，跳过向量索引构建")
            return
        
        if self.model is None:
            logger.error("模型未初始化")
            return
        
        logger.info("🔨 开始构建语义向量索引...")
        
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
            # 组合规则的多个字段（更丰富的语义信息）
            text = f"{rule.description} {' '.join(rule.keywords)} {' '.join(rule.positive_examples[:2])}"
            corpus.append(text)
            self.rule_index.append(item)
        
        # 使用 BGE 模型进行向量化（批量处理）
        logger.info(f"   正在向量化 {len(corpus)} 条规则...")
        self.rule_vectors = self.model.encode(
            corpus,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True  # 归一化，便于计算余弦相似度
        )
        
        # 构建 FAISS 索引（可选，用于大规模检索加速）
        if self.use_faiss:
            dimension = self.rule_vectors.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # 内积索引（归一化后等价于余弦相似度）
            self.faiss_index.add(self.rule_vectors.astype('float32'))
            logger.info(f"✅ FAISS 索引构建完成")
        
        logger.info(f"✅ 语义向量索引构建完成: {len(self.rule_index)} 条规则")
    
    def retrieve_relevant_rules(
        self,
        text: str,
        protocol_id: Optional[str] = None,
        top_k: int = 3,
        use_hybrid: bool = True,
        min_similarity: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        检索相关规则（混合检索：语义 + 关键词）
        
        Args:
            text: 待检索文本
            protocol_id: 指定协议ID（如果为空则检索所有）
            top_k: 返回前 k 个最相关的规则
            use_hybrid: 是否使用混合检索（语义+关键词）
            min_similarity: 最小相似度阈值
        
        Returns:
            相关规则列表
        """
        if self.rule_vectors is None or not self.rule_index:
            logger.warning("❌ 向量索引未构建，返回空结果")
            return []
        
        logger.debug(f"🔍 {'混合' if use_hybrid else '语义'}检索: 文本='{text[:50]}...', 协议={protocol_id}")
        
        # 向量化查询文本
        query_vector = self.model.encode(
            [text],
            normalize_embeddings=True
        )[0]
        
        # 如果指定了协议，先过滤
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
            semantic_similarities = np.dot(protocol_vectors, query_vector)  # 余弦相似度（已归一化）
            
            # 混合检索：结合关键词匹配
            if use_hybrid:
                keyword_scores = np.array([
                    self._keyword_match_score(text, self.rule_index[idx]["rule"])
                    for idx in protocol_indices
                ])
                
                # 融合分数（语义 70% + 关键词 30%）
                final_scores = 0.7 * semantic_similarities + 0.3 * keyword_scores
                logger.debug(f"   使用混合检索（语义 70% + 关键词 30%）")
            else:
                final_scores = semantic_similarities
            
            # 获取 top-k（扩大候选集，后续过滤）
            candidate_k = min(top_k * 2, len(final_scores))
            top_local_indices = np.argsort(final_scores)[-candidate_k:][::-1]
            top_indices = [protocol_indices[i] for i in top_local_indices]
            top_similarities = semantic_similarities[top_local_indices]
            top_scores = final_scores[top_local_indices]
        else:
            # 检索所有规则
            if self.use_faiss and self.faiss_index:
                # 使用 FAISS 加速检索
                candidate_k = min(top_k * 2, len(self.rule_index))
                similarities, top_indices = self.faiss_index.search(
                    query_vector.reshape(1, -1).astype('float32'),
                    candidate_k
                )
                top_similarities = similarities[0]
                top_indices = top_indices[0]
                top_scores = top_similarities  # 暂不支持全局混合检索
            else:
                # 直接计算余弦相似度
                semantic_similarities = np.dot(self.rule_vectors, query_vector)
                
                if use_hybrid:
                    keyword_scores = np.array([
                        self._keyword_match_score(text, item["rule"])
                        for item in self.rule_index
                    ])
                    final_scores = 0.7 * semantic_similarities + 0.3 * keyword_scores
                else:
                    final_scores = semantic_similarities
                
                candidate_k = min(top_k * 2, len(final_scores))
                top_indices = np.argsort(final_scores)[-candidate_k:][::-1]
                top_similarities = semantic_similarities[top_indices]
                top_scores = final_scores[top_indices]
        
        # 构造结果（应用相似度阈值）
        results = []
        for idx, similarity, score in zip(top_indices, top_similarities, top_scores):
            # 应用动态阈值
            item = self.rule_index[idx]
            rule = item["rule"]
            
            # 根据规则类型调整阈值
            adjusted_threshold = self._get_adaptive_threshold(rule.check_type, min_similarity)
            
            if similarity >= adjusted_threshold:
                results.append({
                    "rule_id": rule.rule_id,
                    "category": item["category"],
                    "description": rule.description,
                    "check_type": rule.check_type,
                    "keywords": rule.keywords,
                    "positive_examples": rule.positive_examples,
                    "negative_examples": rule.negative_examples,
                    "severity": rule.severity,
                    "similarity": float(similarity),
                    "hybrid_score": float(score) if use_hybrid else float(similarity)
                })
                
                logger.debug(f"   ✅ 规则 {rule.rule_id}: {rule.description[:30]}... (语义: {similarity:.3f}, 综合: {score:.3f})")
            else:
                logger.debug(f"   ⚠️  规则 {rule.rule_id} 相似度过低 ({similarity:.3f} < {adjusted_threshold:.3f})，已过滤")
            
            # 只返回 top_k 个
            if len(results) >= top_k:
                break
        
        if not results:
            logger.warning(f"❌ 没有检索到任何规则（阈值: {min_similarity}）")
        else:
            logger.info(f"✅ 检索到 {len(results)} 条相关规则")
        
        return results
    
    def _keyword_match_score(self, text: str, rule: Rule) -> float:
        """
        计算关键词匹配分数
        
        Args:
            text: 待检索文本
            rule: 规则对象
        
        Returns:
            匹配分数 [0, 1]
        """
        if not rule.keywords:
            return 0.0
        
        text_lower = text.lower()
        matched = sum(1 for keyword in rule.keywords if keyword.lower() in text_lower)
        
        return matched / len(rule.keywords)
    
    def _get_adaptive_threshold(self, check_type: str, base_threshold: float) -> float:
        """
        根据规则类型动态调整阈值
        
        Args:
            check_type: 规则类型（format/semantic/structure）
            base_threshold: 基础阈值
        
        Returns:
            调整后的阈值
        """
        # format 类规则（格式检查）要求更严格
        if check_type == "format":
            return base_threshold + 0.1
        # semantic 类规则（语义检查）可以宽松一些
        elif check_type == "semantic":
            return base_threshold - 0.05
        # structure 类规则（结构检查）标准阈值
        else:
            return base_threshold
    
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
    
    def save_index(self, file_path: str = "standards/embeddings/index_v2.pkl"):
        """保存向量索引（加速启动）"""
        try:
            save_dir = Path(file_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'wb') as f:
                pickle.dump({
                    "rule_vectors": self.rule_vectors,
                    "rule_index": self.rule_index,
                    "model_name": self.model_name
                }, f)
            
            # 保存 FAISS 索引
            if self.use_faiss and self.faiss_index:
                faiss_path = str(file_path).replace('.pkl', '.faiss')
                faiss.write_index(self.faiss_index, faiss_path)
                logger.info(f"FAISS 索引已保存: {faiss_path}")
            
            logger.info(f"向量索引已保存: {file_path}")
        except Exception as e:
            logger.error(f"保存向量索引失败: {e}")
    
    def load_index(self, file_path: str = "standards/embeddings/index_v2.pkl"):
        """加载向量索引（跳过模型加载和向量化）"""
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                self.rule_vectors = data["rule_vectors"]
                self.rule_index = data["rule_index"]
                saved_model_name = data.get("model_name")
                
                if saved_model_name != self.model_name:
                    logger.warning(f"索引使用的模型 ({saved_model_name}) 与当前模型 ({self.model_name}) 不同")
            
            # 加载 FAISS 索引
            if self.use_faiss:
                faiss_path = str(file_path).replace('.pkl', '.faiss')
                if Path(faiss_path).exists():
                    self.faiss_index = faiss.read_index(faiss_path)
                    logger.info(f"FAISS 索引已加载: {faiss_path}")
            
            logger.info(f"向量索引已加载: {file_path}")
        except Exception as e:
            logger.error(f"加载向量索引失败: {e}")
            logger.info("将重新构建索引...")
            self._build_vector_index()

