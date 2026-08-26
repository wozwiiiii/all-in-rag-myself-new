# Smart-Emergency-Response 代码审查报告

**审查日期**: 2026-08-17  
**项目位置**: `my-rag-try/Smart-Emergency-Response/`  
**审查范围**: `app/`（生产代码）、`tests/`、`eval/`、基础设施文件  
**测试结果**: 64/64 passed ✅ | 真实调用回归 2 轮收敛 ✅  

---

## 一、项目概况

竞赛原型（`baseline/`、`advance/`）→ 企业级 REST API（`app/`），完整保留了竞赛代码用于 B 榜复现。

| 维度 | 竞赛原型 | 生产版 |
|---|---|---|
| 架构 | 单脚本 | FastAPI 分层（core/llm/rag/tabular/orchestrator/api） |
| 安全 | `exec()` + 注入 `os` → 任意命令执行 | 三层沙箱（AST + curated namespace + 进程隔离） |
| 可靠性 | 无重试/无超时 | 重试+deadline+降级+统一错误体 |
| 可观测 | print | JSON 结构化日志 + request_id 贯穿 |
| 部署 | 无 | Dockerfile（多阶段/非root）+ docker-compose + 健康检查 |
| 质量 | 无 | 64 项测试 + 18 题评测集 |

---

## 二、审查发现与修复

### 🔴 高优先级（3 项，全部已修复）

#### H1. 沙箱逃逸：np.memmap 任意文件读取 + 赋值别名链绕过
- **根因**: 原方案用属性黑名单拦截危险 API。实测发现两条绕过路径：
  1. `np.memmap('/etc/shadow', ...)` — `memmap` 未在黑名单中
  2. `c = pd.io.common; o = c.os; o.environ` — 别名赋值链绕过单行属性检查
- **修复**: 核心改为 **curated namespace**。子进程中的 `pd`/`np` 不是模块本体，是 `types.SimpleNamespace(**curated_members)`，仅暴露分析安全 API（DataFrame、merge、mean、linalg.norm 等）。`pd.io`、`np.memmap`、`np.core` 物理不存在。
- **验证**: 新增 26 条攻击载荷全部拦截；合法分析代码（含 `np.linalg.norm`、`.str.contains`、`try/except`）正常通过。

#### H2. Pipeline 超时守卫不完整
- **根因**: 路由分类无超时（LLM 挂起 → 整条请求卡死）；`_gather_first_error` 仅捕获 `AppError`，`asyncio.TimeoutError` 等漏出。
- **修复**: 路由包在 `asyncio.wait_for(deadline)` 内；guard 改为 `except Exception`。
- **文件**: `app/orchestrator/pipeline.py`

#### H3. LLM 调用异常泄漏 + SQLite 连接泄漏
- **根因**: 所有调用方直接调 `.chat()`，重试耗尽后泄漏原始 SDK 异常（含 API base、key）到错误体；SQLite 连接无 `closing`。
- **修复**: 新增 `chat_safe()` 统一包装为 `LLMError`；`_connect()` 用 `contextlib.closing`；5xx 加入重试集合。
- **文件**: `app/llm/client.py`

### 🟡 中优先级（4 项，全部已修复）

#### M1. ReAct 解析器吞尾 + 大小写 + 末轮浪费执行
- **修复**: `_ACTION_RE` 加入前瞻锚（`(?=\nThought:|Action:|Observation:)`）；`action` 统一 `.lower()`；末轮 `force_finish` 禁止代码执行。
- **文件**: `app/tabular/agent.py`

#### M2. 429 Retry-After 未设置 HTTP 头
- **修复**: `RateLimitError` 动态挂 `retry_after` 属性；`app_error_handler` 读取并设 `Retry-After` 头。

#### M3. 死配置项
- **修复**: 移除 `router_max_attempts`、`llm_max_retries`、`llm_retry_wait_s`（代码中硬编码，从未读取这些配置值）。

#### M4. RAG 生成器拒答误判
- **修复**: 拒答标记匹配限定在答案前 60 字符，避免正文自然出现"未提及"等词被误标。
- **文件**: `app/rag/generator.py`

### 🟢 低优先级 / 已知限制（如实记录在 README）

| 项目 | 状态 | 说明 |
|---|---|---|
| `asyncio.to_thread` 不可取消 | 已记录 | 线程启动后 deadline 到达无法中断，需 bounded ThreadPoolExecutor |
| 限流为单实例内存版 | 已记录 | 多实例需 Redis，文档注明 |
| `.wps` 格式无法解析 | 已记录 | ingest 时记录并跳过 |
| 审计日志不持久化 | 已记录 | 当前仅 stdout/file，生产需 ELK |
| 子进程 OS 级降权 | 已记录 | 当前 Docker 非 root 足够，国家级威胁需 seccomp/AppArmor |
| `to_thread` 线程池无界 | 已记录 | 高并发下可能线程爆炸 |

---

## 三、数据目录影响分析

### 3.1 `data/` 目录（课程数据，671 个文件）

```
data/
├── C1/  (1 文件)   — 强化学习 markdown
├── C2/  (2 文件)   — RAG PDF + 蜜蜂养殖 TXT
├── C3/  (11 文件)  — IPCC PDF + 龙/电影 xlsx + 图片
├── C4/  (2 文件)   — 龙 metadata JSON + AI TXT
├── C8/  (651 文件) — 菜谱 markdown+图片（中餐菜谱知识库）
├── C9/  (4 文件)   — Neo4j 图数据（cypher + CSV）
```

**结论：对 Smart-Emergency-Response 零影响。** 这是 DataWhale 秋训营各章教学数据，与危险化学品安全应急竞赛无关。项目代码未引用此目录。

### 3.2 项目自身数据（竞赛数据）

```
Smart-Emergency-Response/
├── datas/文本数据/         — 36 个文件（doc/docx/pdf/wps），应急法规文件
├── advance/datas/
│   ├── 文本数据/            — 同上（baseline 引用的副本）
│   └── 表数据+表结构说明+表介绍/ — 17 个 xlsx 表 + 表结构 docx
└── storage/
    ├── parquet/             — 17 个 .parquet（已转换）
    ├── chroma/              — 向量索引（已构建）
    └── cache/               — embedding 缓存
```

**数据链路完整**：xlsx → parquet（17 表已转换）✅；文本 → chroma（已构建）✅。Config 路径 `PROJECT_ROOT / "advance" / "datas" / ...` 正确指向数据源。

### 3.3 重建判定

**不需要重建。** 数据源未变，parquet 和 chroma 索引均为最新。以下情况需要重建：
- xlsx 源数据更新 → 重新运行 parquet 预转换
- 文本数据增减 → `python -m scripts.build_index`
- embedding 模型更换 → 清空 `storage/cache/` 后重建索引

---

## 四、文件修改清单

| 文件 | 修改类型 | 说明 |
|---|---|---|
| `app/tabular/agent.py` | 重构 | ReAct 解析器（前瞻锚+大小写+末轮禁执行）+ `chat_safe` 调用 |
| `app/tabular/sandbox.py` | 重构 | curated namespace（`_build_pd_namespace`/`_build_np_namespace`）+ 扩展黑名单 |
| `app/llm/client.py` | 重构 | `chat_safe()` 包装 + `_cache_put_many()` 批量写 + 5xx 重试 + `contextlib.closing` |
| `app/orchestrator/pipeline.py` | 修复 | 路由超时 + guard 扩 `Exception` + deadline 传递 |
| `app/api/security.py` | 修复 | `hmac.compare_digest` + 429 Retry-After + 匿名按 IP 限流 |
| `app/api/routes.py` | 修复 | 移除死代码（request_id 降级逻辑 + uuid import） |
| `app/main.py` | 修复 | error handler 输出 Retry-After HTTP 头 |
| `app/core/config.py` | 修复 | 移除死配置项 |
| `app/rag/generator.py` | 修复 | 拒答限定前 60 字符 |
| `app/rag/ingest.py` | 修复 | 移除冗余 force 判断（幂等构建始终先删旧） |
| `tests/test_router.py` | 修复 | fake LLM mock 改 `chat_safe` |
| `README.md` | 更新 | 安全模型描述（三层防护）+ 测试数 64 + 红队审查说明 |

---

## 五、测试覆盖

```
tests/test_sandbox.py  — 26 攻击载荷全拦截 + 6 合法代码通过 + 3 工具测试
tests/test_chunking.py — 标题层级分块 + 长度合并 + 空输入
tests/test_router.py   — LLM 分类 + fenced JSON + heuristic 降级
tests/test_api.py      — 认证 + 限流 + 错误体 + 健康检查
tests/test_config.py   — 默认值 + 环境覆盖 + 校验

64 passed, 0 failed, 14.87s
```

真实调用回归：表格链路 "2025年3月22日开停车企业数量？" → 2 轮收敛，20s 响应 ✅

---

## 六、项目结构（当前状态）

```
Smart-Emergency-Response/
├── app/
│   ├── main.py               # FastAPI 入口
│   ├── core/                  # config / logging / errors
│   ├── llm/client.py          # OpenAI 兼容 + 重试 + 缓存
│   ├── rag/                   # chunking / ingest / retriever / generator
│   ├── tabular/               # sandbox(★) / schema / agent / loader
│   ├── orchestrator/          # router / pipeline
│   └── api/                   # routes / security / schemas
├── tests/                     # 64 项测试
├── eval/                      # 18 题评测 + 报告
├── scripts/build_index.py     # 索引构建 CLI
├── baseline/                  # 竞赛 A 榜（原样保留）
├── advance/                   # 竞赛 B 榜（原样保留）
├── storage/                   # parquet(17) / chroma / cache
├── datas/                     # 竞赛文本数据(36 文件)
├── Dockerfile / docker-compose.yml
├── pyproject.toml / requirements.txt
└── README.md
```
