import os
from unstructured.partition.auto import partition

# PDF文件路径
pdf_path = "../../data/C2/pdf/rag.pdf"

# 使用Unstructured加载并解析PDF文档
elements = partition(
    filename=pdf_path,
    include_metadata=True,
    strategy="hi_res"  # 高分辨率处理，适用于PDF
)

# 打印基本信息
print(f"文档解析完成，共识别出 {len(elements)} 个元素")
print(f"总字符数: {sum(len(str(element)) for element in elements)}")

# 显示前5个元素的内容
print("\n前5个元素内容：")
for i, element in enumerate(elements[:5]):
    print(f"\n元素 {i+1} - 类型: {element.category}")
    print(f"内容: {str(element)}") 