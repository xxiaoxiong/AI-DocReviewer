"""
简单测试：验证 BGE 模型是否能正常加载
"""
import os

# 设置镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

print("=" * 60)
print("🧪 测试 BGE 模型加载")
print("=" * 60)

try:
    print("\n1️⃣ 导入 sentence-transformers...")
    from sentence_transformers import SentenceTransformer
    print("   ✅ 导入成功")
    
    print("\n2️⃣ 加载模型...")
    print("   模型: BAAI/bge-small-zh-v1.5")
    model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    print("   ✅ 模型加载成功")
    
    print("\n3️⃣ 测试向量化...")
    test_text = "这是一个测试文本"
    embedding = model.encode(test_text)
    print(f"   ✅ 向量化成功")
    print(f"   文本: {test_text}")
    print(f"   向量维度: {len(embedding)}")
    print(f"   向量前5个值: {embedding[:5]}")
    
    print("\n4️⃣ 测试批量向量化...")
    texts = ["文档标题", "日期格式", "表格设计"]
    embeddings = model.encode(texts)
    print(f"   ✅ 批量向量化成功")
    print(f"   处理了 {len(texts)} 个文本")
    print(f"   输出形状: {embeddings.shape}")
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！BGE 模型工作正常")
    print("=" * 60)
    
except ImportError as e:
    print(f"\n❌ 导入失败: {e}")
    print("\n💡 解决方案:")
    print("   pip install sentence-transformers torch")
    
except Exception as e:
    print(f"\n❌ 测试失败: {e}")
    print(f"\n错误类型: {type(e).__name__}")
    
    import traceback
    print("\n详细错误信息:")
    traceback.print_exc()
    
    print("\n💡 可能的原因:")
    print("   1. 模型文件损坏，尝试删除缓存重新下载")
    print("   2. 缓存路径: C:\\Users\\你的用户名\\.cache\\huggingface\\hub")
    print("   3. torch 版本不兼容")

