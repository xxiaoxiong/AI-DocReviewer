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
    except Exception as e:
        logger.warning(f"⚠️ 语义检索引擎加载失败，回退到 TF-IDF: {e}")
        rag_engine = RAGEngine(standards_dir=str(standards_dir))
else:
    logger.info("使用传统检索引擎 (TF-IDF)")
    rag_engine = RAGEngine(standards_dir=str(standards_dir))

llm_service = LLMService(use_local=False)  # 默认使用 DeepSeek
reviewer = DocumentReviewer(rag_engine, llm_service)


@router.post("/document", response_model=ReviewResult)
async def review_document(
    file: UploadFile = File(..., description="待审核的 Word 文档"),
    protocol_id: str = Form(..., description="使用的协议ID")
):
    """
    审核文档接口
    
    Args:
        file: Word 文档
        protocol_id: 协议ID（如：GB_T_9704_2012）
    
    Returns:
        审核结果
    """
    # 验证文件类型
    if not file.filename.endswith(('.docx', '.doc')):
        raise HTTPException(status_code=400, detail="只支持 Word 文档（.docx, .doc）")
    
    # 保存上传文件
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = upload_dir / file.filename
    
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"文件已保存: {file_path}")
        logger.info(f"使用协议: {protocol_id}")
        
        # 验证协议是否存在
        available_protocols = rag_engine.list_available_protocols()
        protocol_ids = [p['protocol_id'] for p in available_protocols]
        if protocol_id not in protocol_ids:
            raise HTTPException(
                status_code=400, 
                detail=f"协议 {protocol_id} 不存在。可用协议: {', '.join(protocol_ids)}"
            )
        
        # 审核文档
        logger.info("=" * 80)
        logger.info("🚀 开始调用审核器...")
        logger.info("=" * 80)
        
        result = await reviewer.review_document(
            file_path=str(file_path),
            protocol_id=protocol_id
        )
        
        logger.info("=" * 80)
        logger.info(f"✅ 审核完成，发现 {result.total_issues} 个问题")
        logger.info("=" * 80)
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"❌ 审核失败: {e}")
        logger.error("=" * 80)
        logger.exception("详细错误信息:")
        
        import traceback
        error_trace = traceback.format_exc()
        
        # 提供更友好的错误提示
        error_msg = str(e)
        if "API" in error_msg or "key" in error_msg.lower():
            error_msg = f"API 调用失败: {error_msg}\n\n请检查：\n1. DeepSeek API Key 是否正确\n2. 网络连接是否正常\n3. API 配额是否充足"
        elif "timeout" in error_msg.lower():
            error_msg = f"请求超时: {error_msg}\n\n可能原因：\n1. 网络连接不稳定\n2. 文档过大\n3. API 服务响应慢"
        
        raise HTTPException(status_code=500, detail=error_msg)
    
    finally:
        # 清理临时文件（可选）
        if file_path.exists():
            # os.remove(file_path)  # 取消注释以自动删除
            pass


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
    
    try:
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
                
                # 2. 发送初始化信息
                yield f"data: {json.dumps({'type': 'init', 'total_chunks': len(chunks), 'message': f'文档解析完成，共 {len(chunks)} 个段落'}, ensure_ascii=False)}\n\n"
                
                # 3. 逐块审核并实时推送
                for i, chunk in enumerate(chunks):
                    try:
                        # 发送进度
                        yield f"data: {json.dumps({'type': 'progress', 'current': i + 1, 'total': len(chunks), 'message': f'正在审核第 {i+1}/{len(chunks)} 段...'}, ensure_ascii=False)}\n\n"
                        
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
                        yield f"data: {json.dumps({'type': 'error', 'message': f'审核第 {i+1} 段时出错: {str(e)}'}, ensure_ascii=False)}\n\n"
                
                # 4. 去重和生成摘要
                unique_issues = reviewer._deduplicate_issues(all_issues)
                unique_issues.sort(key=lambda x: (x.page or 0, x.position))
                summary = reviewer._generate_summary(unique_issues)
                
                # 5. 发送完成信号
                yield f"data: {json.dumps({'type': 'complete', 'total_issues': len(unique_issues), 'summary': summary, 'message': '审核完成！'}, ensure_ascii=False)}\n\n"
                
                logger.info(f"流式审核完成: 发现 {len(unique_issues)} 个问题")
            
            except Exception as e:
                logger.error(f"流式审核失败: {e}")
                import traceback
                error_trace = traceback.format_exc()
                logger.error(f"详细错误:\n{error_trace}")
                
                yield f"data: {json.dumps({'type': 'error', 'message': f'审核失败: {str(e)}'}, ensure_ascii=False)}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    
    except Exception as e:
        logger.error(f"流式审核初始化失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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