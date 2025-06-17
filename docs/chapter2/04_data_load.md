# 第一节 数据加载

## 一、文档加载器

在RAG系统中，**数据加载**是整个流水线的第一步，也是至关重要的一步。文档加载器负责将各种格式的非结构化文档（如PDF、Word、Markdown、HTML等）转换为程序可以处理的结构化数据。数据加载的质量会直接影响后续的索引构建、检索效果和最终的生成质量。

### 1.1 主要功能

- **文档格式解析**
将不同格式的文档（如PDF、Word、Markdown等）解析为文本内容。

- **元数据提取**
在解析文档内容的同时，提取相关的元数据信息，如文档来源、页码等。

- **统一数据格式**
将解析后的内容转换为统一的数据格式，便于后续处理。

### 1.2 当前主流RAG文档加载器

| 工具名称 | 核心特点 | 适用场景 | 性能表现 |
|---------|---------|---------|---------|
| **PyMuPDF4LLM** | PDF→Markdown转换，OCR+表格识别 | 科研文献、技术手册 | 开源免费，GPU加速 |
| **TextLoader** | 基础文本文件加载 | 纯文本处理 | 轻量高效 |
| **DirectoryLoader** | 批量目录文件处理 | 混合格式文档库 | 支持多格式扩展 |
| **UnstructuredMarkdownLoader** | Markdown结构解析 | 技术文档、博客 | 结构保留良好 |
| **FireCrawlLoader** | 网页内容抓取 | 在线文档、新闻 | 实时内容获取 |
| **LlamaParse** | 深度PDF结构解析 | 法律合同、学术论文 | 解析精度高，商业API |
| **Docling** | 模块化企业级解析 | 企业合同、报告 | IBM生态兼容 |
| **Marker** | PDF→Markdown，GPU加速 | 科研文献、书籍 | 专注PDF转换 |
| **MinerU** | 多模态集成解析 | 学术文献、财务报表 | 集成LayoutLMv3+YOLOv8 |

### 1.3 文件格式支持范围

**主流工具支持的文件格式：**

- **文本文件**：`.txt`, `.md`, `.org`, `.rst`, `.csv`, `.tsv`
- **通用文档**：`.eml`, `.html`, `.xml`
- **Microsoft Office**：`.doc`, `.docx`, `.ppt`, `.pptx`, `.xls`, `.xlsx`, `.rtf`
- **Open Document**：`.odt`, `.odp`, `.ods`
- **电子书**：`.epub`
- **数据格式**：`.json`, `.jsonl`
- **图像**：`.png`, `.jpeg/.jpg`, `.tiff`, `.bmp`, `.gif`
- **自定义格式**：`.custom`, `.web`

## 二、Unstructured库详解

[**Unstructured**](https://docs.unstructured.io/open-source/) 是一个专业的文档处理库，专门设计用于RAG和AI微调场景的非结构化数据预处理。提供了统一的接口来处理多种文档格式，是目前最受欢迎的文档加载解决方案之一。

### 2.1 Unstructured的核心优势

🚀 **格式支持广泛**
- 支持多种文档格式：PDF、Word、Excel、HTML、Markdown等
- 统一的API接口，无需为不同格式编写不同代码

📊 **智能内容解析**
- 自动识别文档结构：标题、段落、表格、列表等
- 保留文档元数据信息

### 2.2 支持的文档元素类型

Unstructured能够识别和分类以下文档元素：

| 元素类型 | 描述 |
|---------|------|
| `Title` | 文档标题 |
| `NarrativeText` | 正文段落 |
| `ListItem` | 列表项 |
| `Table` | 表格 |
| `Image` | 图像 |
| `Formula` | 公式 |

## 三、从LangChain封装到原始Unstructured

在第一章的示例中，我们使用了LangChain的`UnstructuredMarkdownLoader`，它是对Unstructured库的封装。接下来展示如何直接使用Unstructured库，这样可以获得更大的灵活性和控制力。

### 3.1 安装依赖

```bash
cd code/C2
pip install -r requirements.txt
```

### 3.2 代码示例

让我们创建一个简单的示例，专注于文档加载：

```python
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
```

> **完整代码文件**：[`code/C2/01_unstructured_example.py`](../../code/C2/01_unstructured_example.py)

**注意**：运行前请确保 `data/C2/pdf/rag.pdf` 文件存在。

## 五、总结

本章介绍了RAG系统中的文档加载基础：
- 文档加载器的基本功能
- Unstructured库的核心特性
- 从LangChain封装到原始库的迁移

下一章我们将学习文档分块策略的设计与实现。