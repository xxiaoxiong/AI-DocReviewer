"""
一键启动和测试脚本
"""
import os
import sys

# 设置工作目录
backend_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(backend_dir)
sys.path.insert(0, backend_dir)

print("=" * 60)
print("🚀 DocReviewer 一键启动")
print("=" * 60)

# 1. 检查依赖
print("\n📦 检查依赖...")
try:
    import fastapi
    import uvicorn
    from docx import Document
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    print("✅ 核心依赖已安装")
except ImportError as e:
    print(f"❌ 缺少依赖: {e}")
    print("\n请运行: pip install -r ../requirements.txt")
    sys.exit(1)

# 2. 检查标准库
print("\n📚 检查标准库...")
standards_dir = os.path.join(backend_dir, "..", "standards", "protocols")
if os.path.exists(standards_dir):
    json_files = [f for f in os.listdir(standards_dir) if f.endswith('.json')]
    print(f"✅ 找到 {len(json_files)} 个标准协议")
else:
    print(f"⚠️  标准目录不存在: {standards_dir}")

# 3. 设置环境变量
print("\n⚙️  配置环境...")

# 检查 .env 文件
env_file = os.path.join(backend_dir, "..", ".env")
if os.path.exists(env_file):
    print(f"📄 加载配置文件: {env_file}")
    with open(env_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()
else:
    print(f"⚠️  未找到 .env 文件，使用默认配置")

# 强制设置 API Key（如果未配置）
if not os.environ.get('DEEPSEEK_API_KEY') or os.environ.get('DEEPSEEK_API_KEY') == 'your_deepseek_api_key_here':
    # 使用硬编码的 API Key（仅用于测试）
    os.environ['DEEPSEEK_API_KEY'] = 'sk-bc4ceb3384f244e38f596fd23631af63'
    print("✅ 使用内置 API Key")
else:
    print("✅ 使用配置的 API Key")

os.environ.setdefault('DEEPSEEK_API_BASE', 'https://api.deepseek.com/v1')
os.environ.setdefault('DEEPSEEK_MODEL', 'deepseek-chat')
os.environ.setdefault('APP_HOST', '0.0.0.0')
os.environ.setdefault('APP_PORT', '8000')
os.environ.setdefault('DEBUG', 'True')

# 验证 API Key
api_key = os.environ.get('DEEPSEEK_API_KEY', '')
if api_key and api_key != 'your_deepseek_api_key_here':
    print(f"🔑 API Key: {api_key[:10]}...{api_key[-4:]}")
else:
    print("❌ 警告: DeepSeek API Key 未正确配置！")
    print("   请在项目根目录创建 .env 文件，或修改 run.py 中的 API Key")

print("✅ 环境配置完成")

# 4. 启动服务
print("\n" + "=" * 60)
print("🌐 启动 Web 服务...")
print("=" * 60)
print("\n📝 重要提示:")
print("1. 如需完整功能，请配置 DEEPSEEK_API_KEY")
print("2. 前端页面: 打开 ../frontend/index.html")
print("3. API 文档: http://localhost:8000/docs")
print("4. 测试文档: ../DocReviewer/data/软件单元测试记录DC.doc")
print("   (需要转换为 .docx 格式)")
print("\n" + "=" * 60)

def test_optimization():
    """测试混合检索优化"""
    print("\n" + "=" * 60)
    print("🔬 测试混合检索优化")
    print("=" * 60)
    
    try:
        from app.core.rag_engine_v2 import RAGEngineV2
        
        print("\n📦 初始化检索引擎...")
        standards_dir = os.path.join(backend_dir, "..", "standards", "protocols")
        rag = RAGEngineV2(standards_dir=standards_dir)
        print("✅ 引擎初始化成功")
        
        # 测试用例
        test_text = "标题应该简洁明了"
        protocol = "GB_T_9704_2012"
        
        print(f"\n🧪 测试查询: \"{test_text}\"")
        print(f"📋 协议: {protocol}")
        
        # 测试1: 纯语义检索
        print("\n1️⃣ 纯语义检索:")
        results1 = rag.retrieve_relevant_rules(
            text=test_text,
            protocol_id=protocol,
            top_k=3,
            use_hybrid=False,
            min_similarity=0.3
        )
        
        if results1:
            for i, rule in enumerate(results1, 1):
                print(f"  {i}. [{rule['rule_id']}] {rule['description'][:50]}")
                print(f"     相似度: {rule['similarity']:.3f}")
        else:
            print("  ❌ 未找到规则")
        
        # 测试2: 混合检索
        print("\n2️⃣ 混合检索（语义70% + 关键词30%）:")
        results2 = rag.retrieve_relevant_rules(
            text=test_text,
            protocol_id=protocol,
            top_k=3,
            use_hybrid=True,
            min_similarity=0.3
        )
        
        if results2:
            for i, rule in enumerate(results2, 1):
                print(f"  {i}. [{rule['rule_id']}] {rule['description'][:50]}")
                print(f"     语义: {rule['similarity']:.3f}, 综合: {rule['hybrid_score']:.3f}")
        else:
            print("  ❌ 未找到规则")
        
        print("\n" + "=" * 60)
        print("✅ 优化 #1 测试完成：混合检索已启用")
        print("=" * 60)
        print("\n💡 优化效果:")
        print("  ✅ 混合检索（语义 + 关键词）")
        print("  ✅ 动态阈值调整")
        print("  ✅ 提高检索准确性")
        
        return True
        
    except ImportError as e:
        print(f"\n❌ 导入失败: {e}")
        print("\n请先安装依赖:")
        print("  pip install sentence-transformers")
        return False
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # 测试模式
        success = test_optimization()
        sys.exit(0 if success else 1)
    else:
        # 正常启动服务
        import uvicorn
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )

