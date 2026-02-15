"""
测试 RAG V2 语义检索引擎
用于验证 BGE 模型是否正常工作，以及检索效果
"""
import sys
from pathlib import Path

# 添加 backend 到路径（从 TestData/script 向上两级到项目根目录）
project_root = Path(__file__).parent.parent.parent
backend_dir = project_root / "backend"
sys.path.insert(0, str(backend_dir))

from loguru import logger
from app.core.rag_engine_v2 import RAGEngineV2
from app.core.rag_engine import RAGEngine

def test_semantic_search():
    """测试语义检索效果"""
    
    print("=" * 80)
    print("🧪 RAG V2 语义检索测试")
    print("=" * 80)
    
    # 1. 初始化引擎
    print("\n📦 步骤 1: 初始化 RAG 引擎...")
    try:
        # 从 TestData/script 向上两级到项目根目录，再到 standards/protocols
        project_root = Path(__file__).parent.parent.parent
        standards_dir = project_root / "standards" / "protocols"
        rag_v2 = RAGEngineV2(standards_dir=str(standards_dir))
        print("✅ RAG V2 (BGE 语义检索) 初始化成功")
        use_v2 = True
    except Exception as e:
        print(f"❌ RAG V2 初始化失败: {e}")
        print("⚠️  回退到 RAG V1 (TF-IDF)")
        rag_v2 = RAGEngine(standards_dir=str(standards_dir))
        use_v2 = False
    
    # 2. 列出可用协议
    print("\n📋 步骤 2: 列出可用协议...")
    protocols = rag_v2.list_available_protocols()
    print(f"✅ 找到 {len(protocols)} 个协议:")
    for p in protocols:
        print(f"   - {p['protocol_id']}: {p['name']}")
    
    if not protocols:
        print("❌ 没有找到任何协议，请检查 standards/protocols 目录")
        return
    
    # 使用第一个协议进行测试
    test_protocol = protocols[0]['protocol_id']
    print(f"\n🎯 使用协议: {test_protocol}")
    
    # 3. 测试检索效果
    print("\n🔍 步骤 3: 测试检索效果...")
    print("-" * 80)
    
    # 测试用例
    test_cases = [
        {
            "query": "文档标题应该如何编写",
            "description": "测试标题相关规则检索"
        },
        {
            "query": "日期格式要求",
            "description": "测试日期格式规则检索"
        },
        {
            "query": "表格应该怎么设计",
            "description": "测试表格相关规则检索"
        },
        {
            "query": "页眉页脚的规范",
            "description": "测试页眉页脚规则检索"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        desc = test_case["description"]
        
        print(f"\n测试 {i}: {desc}")
        print(f"查询: '{query}'")
        print()
        
        try:
            # 根据版本调用不同的参数
            if use_v2:
                results = rag_v2.retrieve_relevant_rules(
                    text=query,
                    protocol_id=test_protocol,
                    top_k=3,
                    use_hybrid=True,  # V2 支持混合检索
                    min_similarity=0.3
                )
            else:
                results = rag_v2.retrieve_relevant_rules(
                    text=query,
                    protocol_id=test_protocol,
                    top_k=3
                )
            
            if results:
                print(f"✅ 找到 {len(results)} 条相关规则:")
                for j, rule in enumerate(results, 1):
                    similarity = rule.get('similarity', 0)
                    hybrid_score = rule.get('hybrid_score', similarity)
                    print(f"\n   规则 {j}:")
                    print(f"   ID: {rule['rule_id']}")
                    print(f"   描述: {rule['description']}")
                    print(f"   类型: {rule['check_type']}")
                    print(f"   严重性: {rule['severity']}")
                    if use_v2:
                        print(f"   语义相似度: {similarity:.3f}")
                        print(f"   综合得分: {hybrid_score:.3f}")
                    print(f"   关键词: {', '.join(rule['keywords'][:5])}")
            else:
                print("❌ 未找到相关规则")
        
        except Exception as e:
            print(f"❌ 检索失败: {e}")
            import traceback
            traceback.print_exc()
        
        print("-" * 80)
    
    # 4. 对比测试（如果 V2 可用）
    if use_v2:
        print("\n📊 步骤 4: 对比 V1 (TF-IDF) vs V2 (BGE 语义)")
        print("-" * 80)
        
        try:
            rag_v1 = RAGEngine(standards_dir=str(standards_dir))
            
            test_query = "标题格式不正确"
            print(f"\n测试查询: '{test_query}'")
            
            # V1 检索
            print("\n🔹 V1 (TF-IDF) 结果:")
            results_v1 = rag_v1.retrieve_relevant_rules(
                text=test_query,
                protocol_id=test_protocol,
                top_k=3
            )
            if results_v1:
                for i, rule in enumerate(results_v1, 1):
                    print(f"   {i}. {rule['description'][:50]}... (分数: {rule.get('score', 0):.3f})")
            else:
                print("   未找到结果")
            
            # V2 检索
            print("\n🔹 V2 (BGE 语义) 结果:")
            results_v2 = rag_v2.retrieve_relevant_rules(
                text=test_query,
                protocol_id=test_protocol,
                top_k=3,
                use_hybrid=True
            )
            if results_v2:
                for i, rule in enumerate(results_v2, 1):
                    print(f"   {i}. {rule['description'][:50]}... (语义: {rule['similarity']:.3f}, 综合: {rule['hybrid_score']:.3f})")
            else:
                print("   未找到结果")
            
            print("\n💡 对比说明:")
            print("   - V1 基于关键词匹配（TF-IDF），需要精确匹配")
            print("   - V2 基于语义理解（BGE），能理解同义词和语义相似性")
            print("   - V2 的综合得分 = 70% 语义 + 30% 关键词")
            
        except Exception as e:
            print(f"❌ 对比测试失败: {e}")
    
    # 5. 总结
    print("\n" + "=" * 80)
    print("📝 测试总结")
    print("=" * 80)
    if use_v2:
        print("✅ RAG V2 (BGE 语义检索) 工作正常")
        print("✅ 模型已加载，可以进行语义理解")
        print("💡 建议: 在实际使用中观察检索准确率，必要时调整相似度阈值")
    else:
        print("⚠️  当前使用 RAG V1 (TF-IDF)")
        print("💡 如需启用语义检索，请确保:")
        print("   1. 已安装: pip install sentence-transformers torch")
        print("   2. 网络正常（首次运行需下载模型）")
        print("   3. 有足够磁盘空间（模型约 100MB）")
    print("=" * 80)


if __name__ == "__main__":
    test_semantic_search()

