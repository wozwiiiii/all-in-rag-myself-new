#!/usr/bin/env python3
"""
测试智能查询重写功能
"""

import sys
import os
from pathlib import Path

# 添加模块路径
sys.path.append(str(Path(__file__).parent))

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from dotenv import load_dotenv
from rag_modules.generation_integration import GenerationIntegrationModule

# 加载环境变量
load_dotenv()

def test_query_rewrite():
    """测试智能查询重写功能"""
    
    print("🧪 测试智能查询重写功能")
    print("=" * 60)
    
    # 检查API密钥
    if not os.getenv("MOONSHOT_API_KEY"):
        print("❌ 请设置 MOONSHOT_API_KEY 环境变量")
        return
    
    # 初始化生成模块
    print("🤖 初始化生成模块...")
    try:
        generation_module = GenerationIntegrationModule()
        print("✅ 生成模块初始化成功")
    except Exception as e:
        print(f"❌ 生成模块初始化失败: {e}")
        return
    
    # 测试查询列表
    test_queries = [
        # 应该保持原查询的具体查询
        "宫保鸡丁怎么做",
        "红烧肉的制作方法", 
        "蛋炒饭需要什么食材",
        "糖醋排骨的步骤",
        "如何炒菜不粘锅",
        "怎样调制糖醋汁",
        "麻婆豆腐的做法",
        
        # 应该重写的模糊查询
        "做菜",
        "有什么好吃的",
        "推荐个菜",
        "川菜",
        "素菜",
        "简单的",
        "想吃点什么",
        "有饮品推荐吗",
        "荤菜有哪些",
        "什么菜好做"
    ]
    
    print(f"\n📝 开始测试 {len(test_queries)} 个查询...")
    print("-" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i:2d}. 原始查询: '{query}'")
        
        try:
            # 执行查询重写
            rewritten_query = generation_module.query_rewrite(query)
            
            # 判断是否发生了重写
            if rewritten_query == query:
                print(f"    ✅ 保持原查询: '{rewritten_query}'")
                print(f"    💡 分析: 查询已足够具体明确")
            else:
                print(f"    🔄 重写结果: '{rewritten_query}'")
                print(f"    💡 分析: 原查询较为模糊，已优化")
                
        except Exception as e:
            print(f"    ❌ 重写失败: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 测试总结:")
    print("✅ 具体查询应该保持原样")
    print("🔄 模糊查询应该被重写优化")
    print("💡 大模型自主判断查询的具体性")
    print("✅ 智能查询重写测试完成！")

if __name__ == "__main__":
    test_query_rewrite()
