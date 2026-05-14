# 第二节 多模态嵌入

现代 AI 的一项重要突破，是将简单的词向量发展成了能统一理解图文、音视频的复杂系统。这一发展建立在**注意力机制、Transformer 架构和对比学习**等关键技术之上，它们解决了在共享向量空间中对齐不同数据模态的核心挑战。其发展环环相扣：Word2Vec 为 BERT 的上下文理解铺路，而 BERT 又为 CLIP 等模型的跨模态能力奠定了基础。

## 一、为什么需要多模态嵌入？

前面的章节介绍了如何为文本创建向量嵌入。然而，仅有文本的世界是不完整的。现实世界的信息是多模态的，包含图像、音频、视频等。传统的文本嵌入无法理解“那张有红色汽车的图片”这样的查询，因为文本向量和图像向量处于相互隔离的空间，存在一堵“模态墙”。

**多模态嵌入 (Multimodal Embedding)** 的目标正是为了打破这堵墙。其目的是将不同类型的数据（如图像和文本）映射到**同一个共享的向量空间**。在这个统一的空间里，一段描述“一只奔跑的狗”的文字，其向量会非常接近一张真实小狗奔跑的图片向量。

实现这一目标的关键，在于解决 **跨模态对齐 (Cross-modal Alignment)** 的挑战。以对比学习、视觉 Transformer (ViT) 等技术为代表的突破，让模型能够学习到不同模态数据之间的语义关联，最终催生了像 CLIP 这样的模型。

## 二、CLIP 模型浅析

在图文多模态领域，OpenAI 的 **CLIP (Contrastive Language-Image Pre-training)** 是一个很有影响力的模型，它为多模态嵌入定义了一个有效的范式。

CLIP 的架构清晰简洁。它采用**双编码器架构 (Dual-Encoder Architecture)**，包含一个图像编码器和一个文本编码器，分别将图像和文本映射到同一个共享的向量空间中。

![CLIP Architecture](./images/3_2_1.webp)
*图：CLIP 的工作流程。(1) 通过对比学习训练双编码器，对齐图文向量空间。(2)和(3) 展示了如何利用该空间，通过图文相似度匹配实现零样本预测。*

为了让这两个编码器学会“对齐”不同模态的语义，CLIP 在训练时采用了**对比学习 (Contrastive Learning)** 策略。在处理一批图文数据时，模型的目标是：最大化正确图文对的向量相似度，同时最小化所有错误配对的相似度。通过这种“拉近正例，推远负例”的方式，模型从海量数据中学会了将语义相关的图像和文本在向量空间中拉近。

这种大规模的对比学习赋予了 CLIP 有效的**零样本（Zero-shot）识别能力**。它能将一个传统的分类任务，转化为一个“图文检索”问题——例如，要判断一张图片是不是猫，只需计算图片向量与“a photo of a cat”文本向量的相似度即可。这使得 CLIP 无需针对特定任务进行微调，就能实现对视觉概念的泛化理解。

## 三、常用多模态嵌入模型(以bge-visualized-m3为例)

虽然 CLIP 为图文预训练提供了重要基础，但多模态领域的研究迅速发展，涌现了许多针对不同目标和场景进行优化的模型。例如，BLIP 系列专注于提升细粒度的图文理解与生成能力，而 ALIGN 则证明了利用海量噪声数据进行大规模训练的有效性。

在众多优秀的模型中，由北京智源人工智能研究院（BAAI）开发的 **bge-visualized-m3（Visualized-BGE 的 M3 版本）** 是一个很有代表性的现代多模态嵌入模型。它是在 **BGE-M3**（文本嵌入底座）的基础上引入图像能力而来，体现了当前技术向“更统一、更全面”发展的趋势。

bge-visualized-m3 的核心特性也可以概括为“M3”（主要继承自其文本底座 BGE-M3）：
- **多语言性 (Multi-Linguality)**：支持超过 100 种语言的文本表示，可用于跨语言的图文检索（文本侧）。
- **多功能性 (Multi-Functionality)**：在文本检索场景下，可按需求使用密集检索（Dense Retrieval）、多向量检索（Multi-Vector Retrieval）等不同范式。
- **多粒度性 (Multi-Granularity)**：文本侧可处理从短句到长达 8192 个 token 的长文档，覆盖更广泛的应用需求。

在技术架构上，bge-visualized-m3 会先用视觉编码器提取图像的 **patch token**，再将其映射到与文本同维度的“图像 token”，与文本 token 一起送入 BGE 的 Transformer 编码器进行联合建模，最终得到可用于图文检索的统一向量表示。

## 四、代码示例

### 4.1 环境准备

**步骤1：安装 visual_bge 模块**

```bash
# 进入 visual_bge 目录
cd code/C3/visual_bge

# 安装 visual_bge 模块及其依赖
pip install -e .

# 返回上级目录
cd ..
```

**步骤2：下载模型权重**

```bash
# 运行模型下载脚本
python download_model.py
```

模型下载脚本会自动检查 `../../models/bge/` 目录下是否存在模型文件，如果不存在则从 Hugging Face 镜像站下载。

### 4.2 基础示例

```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from visual_bge.visual_bge.modeling import Visualized_BGE

model = Visualized_BGE(model_name_bge="BAAI/bge-base-en-v1.5",
                      model_weight="../../models/bge/Visualized_base_en_v1.5.pth")
model.eval()

with torch.no_grad():
    text_emb = model.encode(text="datawhale开源组织的logo")
    img_emb_1 = model.encode(image="../../data/C3/imgs/datawhale01.png")
    multi_emb_1 = model.encode(image="../../data/C3/imgs/datawhale01.png", text="datawhale开源组织的logo")
    img_emb_2 = model.encode(image="../../data/C3/imgs/datawhale02.png")
    multi_emb_2 = model.encode(image="../../data/C3/imgs/datawhale02.png", text="datawhale开源组织的logo")

# 计算相似度
sim_1 = img_emb_1 @ img_emb_2.T
sim_2 = img_emb_1 @ multi_emb_1.T
sim_3 = text_emb @ multi_emb_1.T
sim_4 = multi_emb_1 @ multi_emb_2.T

print("=== 相似度计算结果 ===")
print(f"纯图像 vs 纯图像: {sim_1}")
print(f"图文结合1 vs 纯图像: {sim_2}")
print(f"图文结合1 vs 纯文本: {sim_3}")
print(f"图文结合1 vs 图文结合2: {sim_4}")
```

**代码解读：**

- **模型架构**: `Visualized_BGE` 是通过将图像token嵌入集成到BGE文本嵌入框架中构建的通用多模态嵌入模型，具备处理超越纯文本的多模态数据的灵活性。
- **模型参数**:
  - `model_name_bge`: 指定底层BGE文本嵌入模型，继承其强大的文本表示能力。
  - `model_weight`: Visual BGE的预训练权重文件，包含视觉编码器参数。
- **多模态编码能力**: Visual BGE提供了编码多模态数据的多样性，支持纯文本、纯图像或图文组合的格式：
  - **纯文本编码**: 保持原始BGE模型的强大文本嵌入能力。
  - **纯图像编码**: 使用基于EVA-CLIP的视觉编码器处理图像。
  - **图文联合编码**: 将图像和文本特征融合到统一的向量空间。
- **应用场景**: 主要用于混合模态检索任务，包括多模态知识检索、组合图像检索、多模态查询的知识检索等。
- **相似度计算**: 使用矩阵乘法计算余弦相似度，所有嵌入向量都被标准化到单位长度，确保相似度值在合理范围内。

**运行结果：**

```bash
=== 相似度计算结果 ===
纯图像 vs 纯图像: tensor([[0.8318]])
图文结合1 vs 纯图像: tensor([[0.8291]])
图文结合1 vs 纯文本: tensor([[0.7627]])
图文结合1 vs 图文结合2: tensor([[0.9058]])
```

> [完整代码](https://github.com/datawhalechina/all-in-rag/blob/main/code/C3/01_bge_visualized.py)

## 五、前沿模型补充（Gemini Embedding 2 与 jina-embeddings-v5-omni）

前面我们讲了 CLIP 和 Visualized-BGE。最近两年，多模态嵌入又往前走了一步：不仅支持图文，还开始把音频、视频和文档一起纳入统一向量空间。这里用 **Gemini Embedding 2** 和 **jina-embeddings-v5-omni** 做一个实战视角的补充。

### 5.1 jina-embeddings-v5-omni 的特点

jina v5-omni 的核心思路是：在多模态扩展时尽量保留文本底座能力。公开资料里，它采用“冻结编码器 + 轻量投影层”的路线，即文本、视觉、音频主干保持冻结，只训练中间 projector 做跨模态对齐。视频一般按“多帧视觉 + 可选音频”处理，不额外训练一套全新大视频主干。官方口径中，可训练参数约占总参数的 0.35%。

这条路线在论文中被命名为 **GELATO**（*Geometry-preserving Embeddings via Locked Aligned TOwers*）。可以把它理解为“锁住各模态主干、用小连接层做对齐”的模块化方案：相比全量联合训练，训练成本更低；同时文本路径的几何结构被尽量保持，文本输入的 embedding 行为与原有文本底座保持一致，便于平滑升级已有检索/RAG 系统。

![jina-embeddings-v5-omni 架构图](./images/3_2_3.png)
*图：v5-omni 架构。冻结的视觉和音频编码器通过可训练的投影层输入到冻结的文本骨干中。仅训练投影层（占总权重的 0.35%）。任务特定的 LoRA 适配器处理检索、分类、聚类和文本匹配。*

对应的训练基座可以理解为三层映射关系：
- 文本基座：`jina-embeddings-v5-omni-small` 对应 `jina-embeddings-v5-text-small`（Qwen3-0.6B 路径），`jina-embeddings-v5-omni-nano` 对应 `jina-embeddings-v5-text-nano`（EuroBERT-210m 路径）。
- 视觉基座：small 路径使用 Qwen3.5 系视觉编码器，nano 路径使用 SigLIP2 Base 视觉编码器。
- 音频基座：small 与 nano 都基于 Whisper-large-v3 体系（公开资料中也提到与 Qwen2.5-Omni 音频链路相关）。

这套策略的直接收益是文本链路兼容性好、训练与显存开销更低、工程迭代更快；同时它继承了 Matryoshka 维度裁剪能力，便于按场景在效果、存储和延迟之间做权衡。

### 5.2 Gemini Embedding 2 的特点

从 Google 官方资料看，Gemini Embedding 2 的核心特点主要有：

- 原生多模态：文本、图像、音频、视频、PDF 统一映射到同一向量空间。
- 支持交错输入：一条请求里可以混合文本与图像等输入，适合复杂检索场景。
- 维度可调：默认 3072 维，可按成本和效果压缩到更小维度（MRL 思路）。
- 任务前缀可控：可通过 `task` 前缀针对检索、分类、聚类等任务做针对性优化。

工程上，它更像一个“即插即用”的通用多模态嵌入能力，适合先快速搭起统一检索基线。

> [完整代码（Jina v5-omni 示例）](https://github.com/datawhalechina/all-in-rag/blob/main/code/C3/08_jina_embedding_omni.py)

### 5.3 局限与边界

统一多模态表示很强，但不是万能替代方案：

- 强时序视频理解、细粒度音频事件识别这类任务，专用模型往往更稳。
- 如果业务要求强可解释性，单向量方案在“维度级解释”上不如多路检索直观。
- 医疗影像、工业质检等高精度垂直场景，通常仍需领域数据和专门模型配合。

### 5.4 与多向量列关系：互补而非替代

统一多模态嵌入和多向量列在系统里通常是配合使用的：前者主要解决跨模态语义对齐，先把候选快速召回；后者主要解决同一实体在不同语义维度上的独立检索，再结合稀疏/稠密检索做过滤与重排。换句话说，一个负责“先找得到”，另一个负责“排得更准”。

### 5.5 选型建议：以评测与成本约束驱动决策

推荐一个比较稳的推进顺序：

1. 先看目标：你更关心跨模态召回，还是结构化过滤和精排。  
2. 再做评测：在自有数据上比较 Recall@K、nDCG、延迟、吞吐和成本。  
3. 最后算迁移账：现有索引能否复用，Schema 是否要改，是否要重建索引。  

实践里常见做法是：先把统一多模态模型跑通做基线，再按瓶颈逐步加多向量列、重排器和规则过滤，不要一开始就把系统堆太重。

## 练习

尝试把代码中的部分文本替换一下，比如将`datawhale开源组织的logo`替换为`blue whale`看看结果有什么不同。

## 延伸阅读（可选）

- jina-embeddings-v5-omni 技术说明：[https://jina.ai/news/jina-embeddings-v5-omni-multimodal-embeddings-for-text-image-audio-and-video/](https://jina.ai/news/jina-embeddings-v5-omni-multimodal-embeddings-for-text-image-audio-and-video/)
- jina-embeddings-v5-omni 模型集合（Hugging Face）：[https://huggingface.co/collections/jinaai/jina-embeddings-v5-omni](https://huggingface.co/collections/jinaai/jina-embeddings-v5-omni)
- jina Embeddings API：[https://jina.ai/embeddings](https://jina.ai/embeddings)
- jina-embeddings-v5-omni 论文：[https://arxiv.org/abs/2605.08384](https://arxiv.org/abs/2605.08384)

- Gemini Embedding 2 发布说明：[https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-embedding-2/](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-embedding-2/)
- Gemini Embedding 2 模型页（eepMind）：[https://deepmind.google/models/gemini/embedding/](https://deepmind.google/models/gemini/embedding/)
- Gemini Embedding 2（Vertex/Agent Platform 官方文档）：[https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/embedding-2](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/embedding-2)
- Gemini Embedding 2 开发者实践：[https://developers.googleblog.com/building-with-gemini-embedding-2/](https://developers.googleblog.com/building-with-gemini-embedding-2/)
- Gemini API Embeddings 文档：[https://ai.google.dev/gemini-api/docs/embeddings](https://ai.google.dev/gemini-api/docs/embeddings)