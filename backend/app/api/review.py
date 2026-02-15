"""
API 路由 - 文档审核接口
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional
import json
import os
from pathlib import Path
from loguru import logger

from ..core.reviewer import DocumentReviewer
from ..core.rag_engine import RAGEngine
from ..core.rag_engine_v2 import RAGEngineV2
from ..services.llm_service import LLMService
from ..models.document import ReviewResult

router = APIRouter(prefix="/api/review", tags=["审核"])

# 初始化服务（全局单例）
# 获取项目根目录 - 从backend目录向上一级
backend_dir = Path(__file__).parent.parent.parent
project_root = backend_dir.parent
standards_dir = project_root / "standards" / "protocols"

# 🚀 使用新版语义检索引擎（可切换回旧版）
USE_SEMANTIC_SEARCH = True  # 设置为 False 使用旧版 TF-IDF

if USE_SEMANTIC_SEARCH:
    try:
        logger.info("🚀 使用语义检索引擎 V2 (BGE)")
        rag_engine = RAGEngineV2(standards_dir=str(standards_dir))
        
        # 尝试加载已保存的索引
        index_path = project_root / "standards" / "embeddings" / "index_v2.pkl"
        if index_path.exists():
            logger.info("📦 加载已保存的索引...")
            rag_engine.load_index(str(index_path))
        else:
            logger.info("💾 首次启动，保存索引...")
            rag_engine.save_index(str(index_path))
            
    except Exception as e:
        logger.warning(f"⚠️ 语义检索引擎加载失败: {e}")
else:
    logger.info("使用传统检索引擎 (TF-IDF)")
    rag_engine = RAGEngine(standards_dir=str(standards_dir))

llm_service = LLMService(use_local=False)  # 默认使用 DeepSeek
reviewer = DocumentReviewer(rag_engine, llm_service)



@router.post("/document/stream")
async def review_document_stream(
    file: UploadFile = File(...),
    protocol_id: str = Form(...)
):
    """
    流式审核接口（实时返回结果）
    
    适合大文档，可以实时看到审核进度
    """
    if not file.filename.endswith(('.docx', '.doc')):
        raise HTTPException(status_code=400, detail="只支持 Word 文档")
    
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    logger.info(f"流式审核开始: {file_path}, 协议: {protocol_id}")
    
    # 验证协议是否存在
    available_protocols = rag_engine.list_available_protocols()
    protocol_ids = [p['protocol_id'] for p in available_protocols]
    if protocol_id not in protocol_ids:
        raise HTTPException(
            status_code=400, 
            detail=f"协议 {protocol_id} 不存在。可用协议: {', '.join(protocol_ids)}"
        )
    
    async def generate():
        """生成器函数 - 流式推送审核结果"""
        all_issues = []
        
        try:
            # 1. 解析文档
            yield f"data: {json.dumps({'type': 'status', 'message': '正在解析文档...'}, ensure_ascii=False)}\n\n"
            
            doc_structure = reviewer.parser.parse_docx(str(file_path))
            chunks = reviewer.chunker.chunk_by_paragraphs(doc_structure)
            
            logger.info(f"文档分块完成: {len(chunks)} 个块")
            
            # 2. 智能优化过滤
            yield f"data: {json.dumps({'type': 'status', 'message': '正在智能优化审核任务...'}, ensure_ascii=False)}\n\n"
            
            chunks_to_review, optimization_info = reviewer.optimizer.filter_chunks_for_review(
                chunks, protocol_id, rag_engine
            )
            
            # 发送优化信息
            opt_message = f'优化完成：{optimization_info["original_count"]} 个块 -> {optimization_info["final_review_count"]} 个需审核（优化率 {optimization_info["optimization_rate"]:.1f}%）'
            yield f"data: {json.dumps({'type': 'optimization', 'data': optimization_info, 'message': opt_message}, ensure_ascii=False)}\n\n"
            
            # 3. 发送初始化信息
            init_message = f'文档解析完成，共 {len(chunks)} 个段落，需审核 {len(chunks_to_review)} 个'
            yield f"data: {json.dumps({'type': 'init', 'total_chunks': len(chunks), 'chunks_to_review': len(chunks_to_review), 'message': init_message}, ensure_ascii=False)}\n\n"
            
            # 4. 逐块审核并实时推送（只审核需要的块）
            for i, chunk in enumerate(chunks_to_review):
                try:
                    # 发送进度
                    progress_message = f'正在审核第 {i+1}/{len(chunks_to_review)} 段...'
                    yield f"data: {json.dumps({'type': 'progress', 'current': i + 1, 'total': len(chunks_to_review), 'message': progress_message}, ensure_ascii=False)}\n\n"
                    
                    # 审核当前块
                    issues = await reviewer._review_chunk(chunk, protocol_id)
                    
                    # 收集所有问题
                    all_issues.extend(issues)
                    
                    # 如果有问题，立即推送
                    if issues:
                        for issue in issues:
                            yield f"data: {json.dumps({'type': 'issue', 'data': issue.dict()}, ensure_ascii=False)}\n\n"
                    
                except Exception as e:
                    logger.error(f"审核块 {i} 失败: {e}")
                    error_message = f'审核第 {i+1} 段时出错: {str(e)}'
                    yield f"data: {json.dumps({'type': 'error', 'message': error_message}, ensure_ascii=False)}\n\n"
            
            # 5. 处理缓存的结果
            for chunk_id, cached_issues in optimization_info.get('cached_results', {}).items():
                all_issues.extend(cached_issues)
                if cached_issues:
                    for issue in cached_issues:
                        yield f"data: {json.dumps({'type': 'issue', 'data': issue.dict()}, ensure_ascii=False)}\n\n"
            
            # 6. 去重和生成摘要
            unique_issues = reviewer._deduplicate_issues(all_issues)
            unique_issues.sort(key=lambda x: (x.page or 0, x.position))
            summary = reviewer._generate_summary(unique_issues)
            
            # 7. 获取优化统计信息
            optimizer_stats = reviewer.optimizer.get_statistics()
            optimizer_stats['cache_size'] = reviewer.optimizer.get_cache_size()
            
            # 8. 发送完成信号
            yield f"data: {json.dumps({'type': 'complete', 'total_issues': len(unique_issues), 'summary': summary, 'optimization_info': optimization_info, 'optimizer_stats': optimizer_stats, 'message': '审核完成！'}, ensure_ascii=False)}\n\n"
            
            logger.info(f"流式审核完成: 发现 {len(unique_issues)} 个问题，优化率 {optimization_info['optimization_rate']:.1f}%")
        
        except Exception as e:
            logger.error(f"流式审核失败: {e}")
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"详细错误:\n{error_trace}")
            
            error_message = f'审核失败: {str(e)}'
            yield f"data: {json.dumps({'type': 'error', 'message': error_message}, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/protocols")
async def list_protocols():
    """
    列出所有可用的协议
    
    Returns:
        协议列表
    """
    try:
        protocols = rag_engine.list_available_protocols()
        return {
            "total": len(protocols),
            "protocols": protocols
        }
    except Exception as e:
        logger.error(f"获取协议列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/protocols/{protocol_id}/rules")
async def get_protocol_rules(protocol_id: str):
    """
    获取指定协议的所有规则
    
    Args:
        protocol_id: 协议ID
    
    Returns:
        规则列表
    """
    try:
        rules = rag_engine.get_all_rules_by_protocol(protocol_id)
        return {
            "protocol_id": protocol_id,
            "total_rules": len(rules),
            "rules": rules
        }
    except Exception as e:
        logger.error(f"获取协议规则失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs/recent")
async def get_recent_logs(limit: int = 10):
    """
    获取最近的审核日志
    
    Args:
        limit: 返回数量
    
    Returns:
        日志列表
    """
    try:
        from ..core.review_logger import review_logger
        sessions = review_logger.get_recent_sessions(limit)
        return {
            "total": len(sessions),
            "sessions": sessions
        }
    except Exception as e:
        logger.error(f"获取审核日志失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs/{session_id}")
async def get_session_log(session_id: str):
    """
    获取指定会话的详细日志
    
    Args:
        session_id: 会话ID
    
    Returns:
        详细日志
    """
    try:
        from ..core.review_logger import review_logger
        log_file = review_logger.log_dir / f"{session_id}_full.json"
        
        if not log_file.exists():
            raise HTTPException(status_code=404, detail="日志文件不存在")
        
        import json
        with open(log_file, 'r', encoding='utf-8') as f:
            log_data = json.load(f)
        
        return log_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"读取日志失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/document/preview")
async def preview_document_chunks(
    file: UploadFile = File(..., description="待预览的 Word 文档")
):
    """
    预览文档分块情况（不进行审核）
    
    用于调试和可视化分块效果
    
    Returns:
        分块信息
    """
    if not file.filename.endswith(('.docx', '.doc')):
        raise HTTPException(status_code=400, detail="只支持 Word 文档")
    
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename
    
    try:
        # 保存文件
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"预览文档分块: {file_path}")
        
        # 解析文档
        doc_structure = reviewer.parser.parse_docx(str(file_path))
        
        # 分块
        chunks = reviewer.chunker.chunk_by_paragraphs(doc_structure)
        
        # 构造返回数据
        chunks_info = []
        for chunk in chunks:
            chunks_info.append({
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "text_length": len(chunk.text),
                "section": chunk.section,
                "context_before": chunk.context_before,
                "context_after": chunk.context_after,
                "start_pos": chunk.start_pos,
                "end_pos": chunk.end_pos
            })
        
        return {
            "filename": file.filename,
            "total_chunks": len(chunks),
            "total_paragraphs": doc_structure["metadata"]["total_paragraphs"],
            "total_chars": doc_structure["metadata"]["total_chars"],
            "chunks": chunks_info,
            "document_structure": {
                "title": doc_structure["title"],
                "sections": [
                    {
                        "level": s["level"],
                        "title": s["title"],
                        "paragraph_count": len(s["paragraphs"])
                    }
                    for s in doc_structure["sections"]
                ]
            }
        }
    
    except Exception as e:
        logger.error(f"预览失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/document/test-retrieval")
async def test_retrieval(
    text: str = Form(..., description="测试文本"),
    protocol_id: str = Form(..., description="协议ID"),
    top_k: int = Form(5, description="返回数量")
):
    """
    测试RAG检索效果
    
    用于调试和理解检索机制
    
    Returns:
        检索到的规则及相似度
    """
    try:
        logger.info(f"测试检索: 文本='{text[:50]}...', 协议={protocol_id}")
        
        # 检索规则
        relevant_rules = rag_engine.retrieve_relevant_rules(
            text=text,
            protocol_id=protocol_id,
            top_k=top_k,
            use_hybrid=True,
            min_similarity=0.0  # 不过滤，显示所有结果
        )
        
        return {
            "query_text": text,
            "protocol_id": protocol_id,
            "total_matched": len(relevant_rules),
            "rules": relevant_rules
        }
    
    except Exception as e:
        logger.error(f"检索测试失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/vectors/info")
async def get_vector_info():
    """
    获取向量索引信息（证明向量化确实存在）
    
    Returns:
        向量索引的详细信息
    """
    try:
        if rag_engine.rule_vectors is None:
            return {
                "status": "not_initialized",
                "message": "向量索引未初始化"
            }
        
        import numpy as np
        
        # 获取向量信息
        vector_shape = rag_engine.rule_vectors.shape
        vector_dtype = str(rag_engine.rule_vectors.dtype)
        
        # 计算一些统计信息
        vector_mean = float(np.mean(rag_engine.rule_vectors))
        vector_std = float(np.std(rag_engine.rule_vectors))
        vector_min = float(np.min(rag_engine.rule_vectors))
        vector_max = float(np.max(rag_engine.rule_vectors))
        
        # 获取第一条规则的向量作为示例
        first_vector = rag_engine.rule_vectors[0].tolist()[:10]  # 只显示前10维
        
        return {
            "status": "initialized",
            "model_name": rag_engine.model_name,
            "total_rules": len(rag_engine.rule_index),
            "vector_dimension": vector_shape[1],
            "vector_shape": list(vector_shape),
            "vector_dtype": vector_dtype,
            "statistics": {
                "mean": vector_mean,
                "std": vector_std,
                "min": vector_min,
                "max": vector_max
            },
            "sample_vector": {
                "rule_id": rag_engine.rule_index[0]["rule"].rule_id,
                "description": rag_engine.rule_index[0]["rule"].description,
                "vector_preview": first_vector,
                "note": "只显示前10维，完整向量有768维"
            },
            "faiss_enabled": rag_engine.use_faiss and rag_engine.faiss_index is not None
        }
    
    except Exception as e:
        logger.error(f"获取向量信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimizer/clear-cache")
async def clear_optimizer_cache():
    """
    清空优化器缓存
    
    用于释放内存或重置缓存
    
    Returns:
        操作结果
    """
    try:
        old_size = reviewer.optimizer.get_cache_size()
        reviewer.optimizer.clear_cache()
        
        logger.info(f"缓存已清空: 清除了 {old_size} 个缓存项")
        
        return {
            "success": True,
            "message": f"缓存已清空",
            "cleared_items": old_size
        }
    except Exception as e:
        logger.error(f"清空缓存失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/optimizer/stats")
async def get_optimizer_stats():
    """
    获取优化器统计信息
    
    Returns:
        优化器的详细统计信息
    """
    try:
        stats = reviewer.optimizer.get_statistics()
        stats['cache_size'] = reviewer.optimizer.get_cache_size()
        
        return {
            "success": True,
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"获取优化器统计失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))