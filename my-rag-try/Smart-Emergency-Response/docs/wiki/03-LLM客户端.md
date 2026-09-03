# 03 · LLM 客户端（app/llm/client.py）

单一 `LLMClient` 类收口全部 LLM 访问：**chat / rerank / embedding 三类调用 + 统一重试 + SQLite 缓存**。约 190 行，是"基础设施层"的核心。

## 为什么收口（设计决策）

不收口的后果：每个业务模块自己 try/except、自己设超时、自己处理 429——重试策略散落 N 份，行为不可预期，双层重试互相放大（重试风暴）。收口后：

- SDK 客户端显式传 `max_retries=0`（`client.py:44`），重试完全由本类管理
- 重试策略只有一份：`tenacity` 指数退避（2s 起，30s 封顶，最多 4 次）
- 429/5xx 统一映射为内部 `_Retryable429` 再进入重试（`client.py:124-127`）

```python
# client.py — 429 归一化模式（chat 与 _embed_api_call 同构）
except APIStatusError as e:
    if e.status_code == 429 or e.status_code >= 500:
        raise _Retryable429(str(e)) from e   # 可重试
    raise                                    # 4xx 直接抛，不浪费重试
```

## 三类调用接口

| 方法 | 用途 | 降级语义 |
|---|---|---|
| `chat(messages, ...)` | 原始调用，返回 message 对象 | 重试耗尽抛 SDK 原始异常 |
| `chat_safe(messages, ...)` | 业务包装 | 重试耗尽抛 `LLMError`（AppError 子类），上层可降级 |
| `rerank(query, documents, top_n)` | 精排，返回 [(原索引, 分数)] | 同上，由 Retriever 决定降级 |
| `embed_query / embed_documents` | 向量化（带缓存） | 同 chat |

`rerank` 的实现细节：OpenAI SDK **没有封装 `/v1/rerank` 端点**，因此用 `httpx` 直调（`client.py:146-170`）。连接错误统一映射为 `APIConnectionError` 进入 tenacity 重试；非 200 抛 `LLMError`。

> 实测选型数据（2026-08-31）：`BAAI/bge-reranker-v2-m3` 对同域候选的分数分离度（top1-top6 = 0.024）比 `Qwen/Qwen3-Reranker-0.6B`（0.0025）高约 10 倍——前者为默认。

## embedding 磁盘缓存（SQLite）

```python
# client.py 缓存 key：模型名 + 文本 → md5
def _cache_key(self, text: str) -> str:
    raw = f"{self._s.openai_api_embedding_model}:{text}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()
```

设计要点：
- **换模型自动失效**：key 含模型名，旧缓存 miss 而不会误用，无需手动清理
- **存储格式**：`np.float32.tobytes()` 存 BLOB，读回 `np.frombuffer` 零拷贝
- **并发**：`PRAGMA journal_mode=WAL`；批量写 `executemany + INSERT OR REPLACE`
- **批量策略**：先查缓存命中，miss 的按 `embed_batch_size=32` 分批调 API；返回条数与请求校验防错位（`client.py:156-159`）

实测收益：710 分块的增量重建（只改了几篇文档）近乎免费——未变的分块全部命中缓存。

## 一次 chat 的完整生命周期

```
业务模块 → chat_safe()
  → tenacity @retry（APIConnectionError / APITimeoutError / _Retryable429）
    → SDK chat.completions.create（max_retries=0，timeout=45s）
  → 重试耗尽 → LLMError("LLM 调用失败（重试已耗尽）...")
→ 上层：Retriever 降级 RRF / Agent 自愈 / API 层统一错误体
```

`chat_safe` 的另一个价值：**异常信息截断到 200 字符**——SDK 原始异常可能包含 API key（URL 参数回显），直接进错误体会泄漏凭证。

## usage 观测（2026-09-03 引入）

`chat()` 成功后不再丢弃 `resp.usage`，而是累加进客户端实例级计数器并逐笔落日志 `llm_call_usage`（含 model/prompt/completion/total）。

```python
# client.py — 线程安全累加器 + 请求级快照差值
self._usage_lock = threading.Lock()
self._usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}

def usage_snapshot(self) -> dict: ...   # 深拷贝当前累计
def usage_delta(self, before) -> dict:  # 相对某快照的增量
```

**为什么用累加器而非 contextvar**：`generator._extract_points` 用 `ThreadPoolExecutor` 并行抽取，contextvar 在池线程的 copy-context 中写不回父上下文——用 contextvar 会丢并行抽取的 token。实例级累加器 + `threading.Lock` 线程安全、零调用方签名改动；代价是并发请求间 token 归属有交叉误差，观测用途可接受。

`pipeline.answer()` 在请求首尾各取一次快照，差值写入 `QAResult.usage`，随 `/qa` 响应的 `usage` 字段与 `qa_done` 日志返回。实测 text 题 8 次调用 ~6000-8700 tokens——这是后续"抽取轮换小模型降延迟"的成本论证依据。

下一章 → [04-RAG文本检索链路](04-RAG文本检索链路.md)
