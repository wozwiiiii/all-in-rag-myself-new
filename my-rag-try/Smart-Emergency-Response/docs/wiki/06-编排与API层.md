# 06 · 编排与 API 层（app/orchestrator/ + app/api/ + app/main.py）

## 6.1 router.py — 意图路由

**路由是单点，必须永不抛异常**——分类 LLM 挂了也要给出可用路由。

```python
# router.py route()
try:
    content = ...  # LLM 分类（temperature=0, max_tokens=32）
    route = str(json.loads(content).get("route", ""))
    if route in VALID_ROUTES: return route
    raise ValueError(...)
except Exception:
    fallback = _heuristic(question)   # 关键词启发式，永不抛异常
    return fallback
```

- **LLM 路由**：system prompt 定义三路规则（text=政策解读 / table=数量统计 / hybrid=两者），单次调用
- **启发式降级**：`_TABLE_HINTS`（"数量/多少/统计..."）与 `_TEXT_HINTS`（"规定/要求/标准..."）交集判定，无命中默认 hybrid（最保守：两路都跑）
- 分类失败日志 `router_fallback_to_heuristic`（可观测）

> **路由正确率口径修正（2026-09-03）**：Router 只输出 `text/table/hybrid`，**永不输出 `refuse`**——refuse 类 2 题的拒答发生在下游（检索阈值 / 生成拒答），不是路由职责。此前"路由 18/18"把 refuse 题也算入路由正确率是**结构性误计**：refuse 题 `actual=text` vs `expected=refuse` 恒判错。修正后 refuse 题改以 `refused` 标志判定拒答正确率（实测 2/2），其余 16 题路由 16/16。

## 6.2 pipeline.py — 编排

对外异步入口 `answer()`，另有 `answer_stream()` 提供阶段事件流（两者共享 `_answer_stages()` 实现，`answer()` 消费末事件、行为等价）。

### 分支调度

```python
route = await asyncio.wait_for(asyncio.to_thread(self._router.route, question),
                               timeout=deadline)
except asyncio.TimeoutError:
    route = "hybrid"   # 路由超时兜底：最保守两路都跑
```

### hybrid 容错并行（_gather_first_error）

```python
async def guard(coro):
    try:
        return await coro
    except Exception as e:      # 覆盖 SDK 原始异常，不只 AppError
        logger.warning("hybrid_branch_failed", ...)
        return None

r1, r2 = await asyncio.gather(guard(t1), guard(t2))
```

降级语义分层：
- 单路失败 → 幸存路照常返回，`detail` 标记 `degraded: True`（hybrid-partial）
- 一路拒答 → 非拒答路为准（`hybrid-single`，避免"无法回答"融进最终答案）
- 两路成功 → LLM 融合（`chat_safe` 包裹，融合失败退化为表格路答案——数据部分本就以表格为准）
- 两路全挂 → `all_branches_failed` 502

### 总超时

`asyncio.wait_for(120s)` 硬上限，超时抛 `qa_timeout`（504）——防单问题拖垮服务。

### 阶段事件流（SSE，2026-09-03 引入）

`_answer_stages()` 将编排拆为 async generator，在关键节点 yield 事件：

```
{"type":"stage","stage":"routing"}   → 路由判定开始
{"type":"route","route":<text|table|hybrid>}  → 已判定
{"type":"stage","stage":"working","branches":[...]}  → 分支执行中
{"type":"answer", ...QAResult 字段}  → 最终答案（含 usage）
{"type":"error","code","message"}    → 致命错误收尾
```

**设计判断（如实）**：分支（检索/生成/表分析）是单次阻塞调用，无法在内部细分"检索/生成"阶段，故执行期统一发 `working` 事件。选择阶段事件流而非全 token 流，是因为 two_pass 答案主体是程序化拼接的要点（非 token 流），且全 token 流需 `AsyncOpenAI` 破坏 LLM 收口惯例，收益小于侵入。

**证据留存（evidence，2026-09-06 引入）**：`QAResult.evidence` 汇总各分支证据——text 路=召回 chunk 原文（800 字/块）拼接、table 路=沙箱 Observation（见 [05 章](05-表格Agent与安全沙箱.md)）、hybrid 合并两路（表格 Observation 在前，数值以表格路为准）。用途：评测侧事实一致性校验（[07 章](07-评测与可观测.md) 口径 v2）。同步 API 响应体（QAResponse 显式构造）不含该字段，仅评测与 SSE answer 事件可见，体积约 5k 字符。

### 输出侧注入防护（guard.py，2026-09-03 引入）

输入侧结构隔离（agent 选表拆 system/user）之外，`pipeline.answer()` 出口统一做 `check_prompt_leak` 检测：答案命中 prompt 元层特征（角色定义/协议标记/画像标题）即替换为安全文案，日志 `prompt_leak_detected`。

**关键取舍**：特征只收"prompt 元层"措辞，**刻意不收**可合法出现在业务答案里的 SQL 规则（如 `delete_status=0`）。初版误把 SQL 规则当泄露特征，导致全部 table/hybrid 题被误拦（avg 一度跌至 0.5943）——表格路 agent 会把业务规则当筛选依据正常写出，不是泄露。这印证了"安全检测不能与业务答案词汇重叠"的原则（与 sandbox 白名单哲学同源）。

## 6.3 main.py — 应用装配

lifespan 内完成（顺序有讲究）：

1. **配置校验**：缺 `OPENAI_API_KEY` 直接 `ConfigError` 拒绝启动（fail-fast，所有问答路径都依赖 LLM）
2. **Excel→parquet 预转换**（幂等，失败只告警不阻断——table 路可能仍可用旧 parquet）
3. **组件装配**：`LLMClient` + `QAPipeline` 注入 `app.state`

中间件与错误处理：
- `request_context`：X-Request-ID 生成/透传 + 日志 contextvar
- `AppError` → 统一 JSON 体（RateLimitError 附 `Retry-After` 头）
- 未知异常 → 500 通用文案，堆栈只进日志

探针免认证（K8s 惯例），业务接口统一挂认证依赖。

## 6.4 api/security.py — 认证与限流

### X-API-Key 认证

```python
# 常数时间比较，防时序侧信道逐字节探测
if not any(hmac.compare_digest(provided, k) for k in settings.api_key_list):
    raise AuthError(...)
```

多 key 逗号分隔（轮换友好）。**认证关闭时按客户端 IP 限流**——避免匿名用户共享一个桶互相 DoS（日志明示 auth_enabled=false）。

### 滑动窗口限流

```python
# RateLimiter.check()：deque 时间戳 + 线程锁
q = self._hits[key]
while q and q[0] <= now - WINDOW_SECONDS:
    q.popleft()                      # 弹出窗口外
if len(q) >= self._limit:
    retry_after = int(WINDOW_SECONDS - (now - q[0])) + 1
    raise RateLimitError(...)        # handler 转 Retry-After 头
q.append(now)
```

**已知限制（README 声明）**：限流状态在进程内存，单实例有效；多实例需 Redis（接口已收敛在 security.py，替换面小）。

## 6.5 api/routes.py — 路由与探针分离

| 路由 | 语义 | 检查项 |
|---|---|---|
| `POST /api/v1/qa` | 问答（同步） | 认证+限流；2-500 字校验（pydantic schemas）；响应含 `usage` |
| `POST /api/v1/qa/stream` | 问答（SSE 流式） | 认证+限流；事件 `data: {json}\n\n` 编码；异常以 error 事件收尾 |
| `GET /api/v1/health` | **存活**（liveness） | 无——进程活着即返回 |
| `GET /api/v1/ready` | **就绪**（readiness） | llm_configured / vector_index / table_data 三项 |

liveness 与 readiness 分离的原因：存活探针不该因依赖故障重启进程（否则雪崩），就绪探针负责摘除不健康实例。

**SSE 错误语义**：流式响应开始后无法改 HTTP 状态码，因此 `/qa/stream` 内异常由 `event_source` 捕获，以 `error` 事件收尾（而非抛给全局 handler 导致 500 截断）。

pipeline 通过 `app.state` 注入而非全局——测试时可以替换为假 pipeline（test_api.py 即此模式）。

下一章 → [07-评测与可观测](07-评测与可观测.md)
