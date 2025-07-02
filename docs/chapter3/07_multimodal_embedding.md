# 第二节 多模态嵌入

现代 AI 的一项重要突破，是将简单的词向量发展成了能统一理解图文、音视频的复杂系统。这一发展建立在**注意力机制、Transformer 架构和对比学习**等关键技术之上，它们解决了在共享向量空间中对齐不同数据模态的核心挑战。其发展环环相扣：Word2Vec 为 BERT 的上下文理解铺路，而 BERT 又为 CLIP 等模型的跨模态能力奠定了基础。

## 一、为什么需要多模态嵌入？

前面的章节介绍了如何为文本创建向量嵌入。然而，仅有文本的世界是不完整的。现实世界的信息是多模态的，包含图像、音频、视频等。传统的文本嵌入无法理解“那张有红色汽车的图片”这样的查询，因为文本向量和图像向量处于相互隔离的空间，存在一堵“模态墙”。

**多模态嵌入 (Multimodal Embedding)** 的目标正是为了打破这堵墙。它的核心是将不同类型的数据（如图像和文本）映射到**同一个共享的向量空间**。在这个统一的空间里，一段描述“一只奔跑的狗”的文字，其向量会非常接近一张真实小狗奔跑的图片向量。

实现这一目标的关键，在于解决了**跨模态对齐 (Cross-modal Alignment)** 的挑战。以对比学习、视觉 Transformer (ViT) 等技术为代表的突破，让模型能够学习到不同模态数据之间的语义关联，最终催生了像 CLIP 这样的模型。

## 二、CLIP 模型浅析

在图文多模态领域，OpenAI 的 **CLIP (Contrastive Language-Image Pre-training)** 是一个很有影响力的模型，它为多模态嵌入定义了一个有效的范式。

CLIP 的架构清晰简洁。它采用**双编码器架构 (Dual-Encoder Architecture)**，包含一个图像编码器和一个文本编码器，分别将图像和文本映射到同一个共享的向量空间中。

![CLIP Architecture](./images/3_2_1.webp)
*图：CLIP 的工作流程。(1) 通过对比学习训练双编码器，对齐图文向量空间。(2)和(3) 展示了如何利用该空间，通过图文相似度匹配实现零样本预测。*

为了让这两个编码器学会“对齐”不同模态的语义，CLIP 在训练时采用了**对比学习 (Contrastive Learning)** 策略。在处理一批（Batch）图文数据时，模型的目标是：最大化正确图文对的向量相似度，同时最小化所有错误配对的相似度。通过这种“拉近正例，推远负例”的方式，模型从海量数据中学会了将语义相关的图像和文本在向量空间中拉近。

这种大规模的对比学习赋予了 CLIP 有效的**零样本（Zero-shot）识别能力**。它能将一个传统的分类任务，转化为一个“图文检索”问题——例如，要判断一张图片是不是猫，只需计算图片向量与“a photo of a cat”文本向量的相似度即可。这使得 CLIP 无需针对特定任务进行微调，就能实现对视觉概念的泛化理解。

## 三、常用多模态嵌入模型(以bge-visualized-m3为例)

虽然 CLIP 为图文预训练提供了重要基础，但多模态领域的研究迅速发展，涌现了许多针对不同目标和场景进行优化的模型。例如，BLIP 系列专注于提升细粒度的图文理解与生成能力，而 ALIGN 则证明了利用海量噪声数据进行大规模训练的有效性。

在众多优秀的模型中，由北京智源人工智能研究院（BAAI）开发的 **BGE-M3** 是一个极具代表性的现代多模态嵌入模型。它在多语言、多功能和多粒度处理上都表现出色，体现了当前技术向“更统一、更全面”发展的趋势。

BGE-M3 的核心特性可以概括为“M3”：
- **多语言性 (Multi-Linguality)**：原生支持超过 100 种语言的文本与图像处理，能够轻松实现跨语言的图文检索。
- **多功能性 (Multi-Functionality)**：在单一模型内同时支持密集检索（Dense Retrieval）、多向量检索（Multi-Vector Retrieval）和稀疏检索（Sparse Retrieval），为不同应用场景提供了灵活的检索策略。
- **多粒度性 (Multi-Granularity)**：能够有效处理从短句到长达 8192 个 token 的长文档，覆盖了更广泛的应用需求。

在技术架构上，BGE-M3 采用了基于 XLM-RoBERTa 优化的联合编码器，并对视觉处理机制进行了创新。它不同于 CLIP 对整张图进行编码的方式，而是采用**网格嵌入 (Grid-Based Embeddings)**，将图像分割为多个网格单元并独立编码。这种设计显著提升了模型对图像局部细节的捕捉能力，在处理多物体重叠等复杂场景时更具优势。

凭借其统一的多模态、多语言和多功能设计，BGE-M3 为构建全球化、高性能的 RAG 应用提供了一个强大而便捷的解决方案。

## 六、代码实战：使用 LlamaIndex 构建图文 RAG

LlamaIndex 为构建多模态 RAG 提供了强大的支持。下面，我们将演示如何使用 LlamaIndex 和 CLIP 模型，构建一个可以对图像进行提问的简单 RAG 应用。

### 7.1 环境准备

首先，安装必要的库：
```bash
pip install llama-index-multi-modal-llms-openai llama-index-embeddings-clip llama-index-vector-stores-qdrant qdrant-client
```

### 7.2 数据准备

准备一些图像文件，并将它们放在一个文件夹中，例如 `data/C3/images/`。

### 7.3 索引图像

接下来，我们加载图像，使用 CLIP 模型为它们创建嵌入，并将它们索引到 Qdrant 向量数据库中。

```python
import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.clip import ClipEmbedding
import qdrant_client

# 设置 OpenAI API Key
os.environ["OPENAI_API_KEY"] = "sk-..."

# 1. 加载图像数据
image_loader = SimpleDirectoryReader("data/C3/images/")
documents = image_loader.load_data()

# 2. 初始化 Qdrant 向量存储
client = qdrant_client.QdrantClient(path="qdrant_db")
vector_store = QdrantVectorStore(client=client, collection_name="image_collection")

# 3. 初始化 CLIP 嵌入模型
clip_embedding = ClipEmbedding()

# 4. 构建并持久化索引
image_index = VectorStoreIndex.from_documents(
    documents,
    vector_store=vector_store,
    embed_model=clip_embedding,
)

print("图像索引构建完成！")
```

### 7.4 查询

索引构建完成后，我们就可以用文本查询来检索相关的图像，并让多模态 LLM (GPT-4V) 来回答问题。

```python
from llama_index.multi_modal_llms.openai import OpenAIMultiModal

# 1. 从已持久化的索引加载
client = qdrant_client.QdrantClient(path="qdrant_db")
vector_store = QdrantVectorStore(client=client, collection_name="image_collection")
image_index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model=ClipEmbedding() # 加载时也需要指定嵌入模型
)

# 2. 初始化多模态 LLM
openai_mm_llm = OpenAIMultiModal(
    model="gpt-4-vision-preview", 
    max_new_tokens=1000
)

# 3. 创建查询引擎
query_engine = image_index.as_query_engine(
    multi_modal_llm=openai_mm_llm,
    similarity_top_k=2 # 检索最相关的2张图片
)

# 4. 执行查询
query_str = "请描述图片中的车辆是什么颜色的，以及它在什么场景下？"
response = query_engine.query(query_str)

print("问题:", query_str)
print("回答:", response.response)

# 打印引用的源文件
for source_node in response.source_nodes:
    print("- 来源:", source_node.metadata.get('file_path'))
```

## 七、前沿现状与新兴方向

多模态领域正处在一个重要的发展阶段，基础模型的能力得到了显著提升，同时新的研究方向也在不断涌现。

### 8.1 代表性模型

- **GPT-4o / GPT-4V**: 在多模态基准测试中表现卓越，具备强大的图文理解和对话能力。
- **Gemini 系列**: 支持原生视频分析，能够处理长达数小时的音频和数百万 token 的上下文。
- **Claude 3.5 Sonnet**: 在代码和复杂推理任务上领先，并具备强大的视觉分析能力。
- **开源模型**: LLaVA-NeXT、ImageBind 等开源模型也在不断推动技术民主化，它们在特定任务（如高分辨率处理、多模态统一嵌入）上取得了显著进展。

### 8.2 新兴的技术方向

- **上下文内多模态学习 (In-context Multimodal Learning)**：增强模型在少样本（Few-shot）场景下跨模态学习的能力。
- **更先进的训练策略**：包括从概念对齐到指令微调的课程学习（Curriculum Learning）、减少对配对数据依赖的自监督学习，以及用于更好人类对齐的 RLHF。
- **统一的 Transformer 架构**：探索能够处理任意模态组合，同时保持专业能力的通用架构。

### 8.3 核心挑战

尽管取得了巨大进展，多模态嵌入仍面临诸多挑战：

- **模态缺失问题 (Missing Modality)**：当输入中缺少某个模态（如只有文本没有图像）时，系统性能可能会下降。
- **幻觉问题 (Hallucination)**：在跨模态场景下，模型仍可能生成与事实不符的内容。
- **计算成本**：训练和推理的巨大开销限制了其广泛部署。
- **评测困难**：现有基准趋于饱和，缺乏衡量真实世界跨模态能力的标准化框架。

### 本章小结

从 CLIP 的“对齐”哲学，到 BLIP 的“理解与生成”并重，再到 ALIGN 的“规模即力量”以及 BGE-M3 的“多语言统一”，我们可以看到多模态嵌入技术正沿着不同的路径深化。它们共同的目标是打破模态的壁垒，构建一个更统一、更全面的语义表示空间。

在为您的 RAG 应用选择多模态方案时，请将这些模型的特点与您的具体需求相结合：
-   如果您的核心任务是**跨模态检索与分类**，并希望有强大的零样本泛化能力，**CLIP** 及其思想是绝佳的起点。
-   如果应用需要**基于图像进行复杂的问答或生成详细描述**，**BLIP** 系列模型将是更合适的选择。
-   如果您的应用需要服务全球用户，处理**多种语言的图文内容**，**BGE-M3** 则提供了极具吸引力的统一解决方案。

理解这些模型的差异，将帮助您在多模态的世界中做出更明智的决策。

> 这场从静态词向量到动态、上下文感知、跨模态表示的演进，是人工智能短暂历史中最重大的成就之一。它反映了我们向着能够理解和生成人类全部体验模态的 AI 系统迈进的根本性转变，也让我们离真正的人工智能更近了一步。