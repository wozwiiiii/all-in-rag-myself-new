# 04 · RAG 文本检索链路（app/rag/）

五个模块构成完整链路：**chunking（提取分块）→ ingest（索引构建）→ bm25（精确匹配）→ retriever（融合检索）→ generator（生成）**。

```
构建期（scripts/build_index.py，与服务启动分离）
  文档 → extract_file → chunk_blocks → embed_documents(缓存) → Chroma 持久化

查询期（Retriever）
  query → 全量 cosine top-20 ─┐
                             ├→ 并集去重 → 阈值过滤 → rerank 精排(或RRF) → top-6
  query → BM25 top-20 ───────┘
```

## 4.1 chunking.py — 提取与语义分块

### 文档提取（extract_file → extract_docx / extract_pdf）

- **docx**：按段落迭代，`Heading N` 样式识别为标题（level 1-4），表格按行拼接保留结构
- **pdf**：逐页 `pypdf.extract_text()`；**无文本层页面（扫描版）→ OCR 兜底**：

```python
# chunking.py extract_pdf() — 按页检测，扫描页走 OCR
for i, page in enumerate(reader.pages):
    text = (page.extract_text() or "").strip()
    if text:
        blocks.append(Block(level=0, text=text))
    else:
        ocr_text = _ocr_pdf_page(path, i)   # pypdfium2 栅格化(scale=2.0) + RapidOCR
        if ocr_text:
            blocks.append(Block(level=0, text=ocr_text))
```

OCR 兜底的关键设计：
- **依赖可选**：`pip install ".[ocr]"`（约 300MB onnxruntime+opencv）；缺失时告警跳过，不阻断构建
- **引擎单例**：`_OCR_ENGINE` 模块级缓存，模型只加载一次
- **实例验证**：GB30871（29 页扫描版）OCR 后评测题 0.15→0.85

### 分块算法（chunk_blocks）

**标题层级分块**而非固定窗口：遇到标题（level>0）flush 当前段落缓冲，标题入 `heading_stack`；每个 chunk 记录完整标题路径（`"某文件 > 第一章 > 1.2 节"`）——召回片段自带来源章节边界，可被独立引用。

超长 section（>1200 字）二次切分，相邻块保留 150 字重叠（`_merge_overflow`，防语义在边界处断裂）。

## 4.2 ingest.py — 索引构建

```python
# chunk_id 稳定性设计：内容变则 id 必变
raw = f"{chunk.source}|{chunk.heading_path}|{chunk.index}|{md5(chunk.text)[:8]}"
chunk_id = md5(raw)
```

- 构建即全量重建（delete_collection + 重建），幂等；成本被 embedding 缓存摊平
- 单文件解析失败不阻断整体（记录到 `report.skipped`）
- 构建产物全部落 `storage/`（.gitignore 排除），**构建与服务分离**：服务启动只读加载，重启秒级

## 4.3 bm25.py — BM25 精确匹配路（2026-08-31 引入）

dense 检索对专有名词/编号（"GB30871""动火作业"）是弱项——语义空间里编号区分度低。BM25 的词频精确匹配恰好互补。

```python
# bm25.py
def _tokenize(text): return [t for t in jieba.lcut(text, HMM=False) if t.strip()]

class BM25Index:          # 全量 710 chunk 内存索引，零 API 成本，毫秒级查询
    def search(self, query, top_k) -> list[tuple[int, float]]

def rrf_fuse(rankings, *, k=60) -> list[int]:
    # Reciprocal Rank Fusion: score(d) = Σ 1/(k + rank_i(d))
```

**RRF 而非加权分数融合的原因**：BM25 分与向量 cosine 量纲不可比，RRF 只用排名，无需调权。

## 4.4 retriever.py — 融合检索（核心）

### 坐标系设计（混合检索的正确性根基）

启动时一次 `collection.get(include=[documents, metadatas, embeddings])` 拉全量建内存索引：

```python
# retriever.py __init__ — 行归一化后，点积即 cosine
emb = np.asarray(got["embeddings"], dtype=np.float32)
norms = np.linalg.norm(emb, axis=1, keepdims=True)
self._emb_normed = emb / norms
```

向量分与 BM25 分共享同一"全量文档索引"坐标——**避免两路候选无法对齐**的坐标混乱问题。710 chunk 规模下全量矩阵乘（毫秒级）既精确又快于 HNSW 近似；语料到 10^5+ 再切回 Chroma query + id 映射。

### 检索主流程（retrieve）

```
1. 向量路：全量 cosine → top recall_k
2. BM25 路：search → top recall_k（失败降级纯向量）
3. 并集去重 → 向量阈值过滤（拒答语义保持纯向量，BM25 不参与拒答判定）
4. 排序：rerank 精排（失败/关闭 → RRF 融合序）
5. 截断 top_k
```

**阈值语义（重要设计决策）**：BM25-only 候选也必须过向量 cosine ≥ 0.35 才进入池——拒答行为不因加 BM25 而改变（安全回归零风险）。

### 降级链（每层失败都有退路）

| 故障点 | 降级行为 | 日志关键字 |
|---|---|---|
| BM25 依赖缺失/构建失败 | 纯向量路 | `bm25_failed_fallback_vector` |
| BM25 查询异常 | 纯向量路 | `bm25_search_failed` |
| rerank 失败 | RRF 融合序 | `rerank_failed_fallback_rrf` |
| 索引不存在 | 启动报错（fail-fast） | — |

### 实测数据（2026-08-31 诊断脚本）

| 题 | 向量排名 | BM25 排名 | RRF 融合 | 说明 |
|---|---|---|---|---|
| 动火分级（Q2） | 3 | **1** | **1** | 术语场景 BM25 强 |
| 重大隐患判定（Q7） | **1** | 3 | **1** | 语义场景向量强 |
| 受限空间措施（Q3） | 3 | 2 | **2** | 融合自适应取优 |

结论：两路**互补**，RRF 融合后取到更强一路——单路最优的题融合后不劣化，单路弱的题融合后改善。

## 4.5 generator.py — 两遍生成与双层拒答

### 演进史：单遍 → 两遍（三次迭代，2026-09-01）

**问题（诊断实测）**：Q1 丢"感知"、Q3 丢"监护"——证据 rerank 分 0.97+ 仍在上下文中，答案不复现。单遍生成要求模型同时做"逐 chunk 相关性判断"+"800 字内压缩成文"，**压缩有损且丢的正是具体条目**。

**迭代 1（失败）**：全量两遍——第一遍对整个上下文抽要点、第二遍按要点重组。实测：抽取轮自己丢词（Q1 抽 9 条丢感知，Q3 抽 20 条丢监护）——~7k 字符上下文上的**注意力稀释**；且重组轮面对 30+ 要点仍在压缩丢弃，155.8s 逼近超时。

**迭代 2（部分成功）**：map-reduce 抽取——逐 chunk 并行抽取（每路 ≤1.2k 字符，任务平凡化）。实测抽取覆盖 100%，但 LLM 重组轮仍丢词 + 延迟无界（72B 输出 ~9 字/秒，长答案物理上慢）。

**迭代 3（最终）**：**要点程序化拼接为答案主体**——代码拼接零丢失，关键词覆盖由抽取轮**确定性保证**；LLM 仅生成 ≤150 字结论段（输出极小 → 延迟有界）：

```python
# generator.py 两遍生成主流程
points = _extract_points(question, chunks, llm)   # map：逐 chunk 并行抽取（ThreadPoolExecutor）
if points:
    summary = _summarize_points(question, points, llm)  # 结论段 ≤150 字（max_tokens=300）
    answer = f"{summary}\n\n{points}" if summary else points  # 程序化拼接：零丢失
```

抽取轮的工程细节：
- 行级来源标注（行尾 `[i]`）+ 全局重编号（剥离模型局部编号统一 1..N）
- 无关 chunk（输出"无"）不产生要点行
- 单路失败返回空串不阻断；**全失败退化为单遍生成**（原行为兜底）

**实测收益（R8）**：text 关键词覆盖 90%→**100%**（8/8 全部 0.850），总分 **0.8667** 历史最高；Q1 延迟 39→18.6s、Q3 92.5→46.2s（结论段输出远小于长重组答案）。

### 单遍生成（fallback 路径）

prompt 约束 6 条：只依据参考资料 / 引用来源编号 / 宁拒不编 / 先结论后依据 / 列举类按条覆盖 / 800 字上限（宁简勿缺条目）。`two_pass=False` 或抽取全失败时走此路。

### 双层拒答

```python
# 第一层：检索为空 → 固定话术拒答，不调用 LLM（省 token 且确定性强）
if not chunks:
    return REFUSAL_ANSWER, [], True

# 第二层：模型自行拒答检测——限定答案前 60 字符匹配拒话术标记
head = answer[:60]
model_refused = any(m in head for m in _REFUSAL_MARKERS)
```

两遍路径下第二层作用于"结论段 + 要点开头"；正文自然出现"未提及"不误判（AUDIT M4 修复，有专门回归测试）。

sources 响应携带 `score`（向量分）与可选 `rerank_score`——每个证据的相关性可审计。

**证据原文留存（2026-09-06 引入）**：text 路召回的 chunk 原文（800 字/块）经 `QAResult.evidence` 透传（[06 章](06-编排与API层.md)），供评测口径 v2 的证据一致性核对——答案中的数字是否真出自召回文本，可审计而非自评。

下一章 → [05-表格Agent与安全沙箱](05-表格Agent与安全沙箱.md)
