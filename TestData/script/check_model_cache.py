"""
检查 BGE 模型缓存是否存在
"""
import os
from pathlib import Path

cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
model_dir = cache_dir / "models--BAAI--bge-small-zh-v1.5"

print("=" * 60)
print("🔍 检查 BGE 模型缓存")
print("=" * 60)
print(f"\n缓存目录: {cache_dir}")
print(f"模型目录: {model_dir}")
print()

if model_dir.exists():
    print("✅ 模型目录存在")
    print("\n目录内容:")
    for item in model_dir.rglob("*"):
        if item.is_file():
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"   {item.relative_to(model_dir)} ({size_mb:.2f} MB)")
    print("\n✅ 模型已下载，可以离线使用")
else:
    print("❌ 模型目录不存在")
    print("\n💡 可能的原因:")
    print("   1. 模型还没下载")
    print("   2. 下载到了其他位置")
    print("\n尝试查找其他可能的位置...")
    
    # 检查其他可能的缓存位置
    other_locations = [
        Path.home() / ".cache" / "torch" / "sentence_transformers",
        Path.home() / ".cache" / "huggingface",
    ]
    
    for loc in other_locations:
        if loc.exists():
            print(f"\n找到缓存目录: {loc}")
            for item in loc.rglob("bge*"):
                print(f"   {item}")

print("\n" + "=" * 60)

