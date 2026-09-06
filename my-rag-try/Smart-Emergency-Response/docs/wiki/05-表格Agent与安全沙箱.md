# 05 · 表格 Agent 与安全沙箱（app/tabular/）

四个模块：**loader（xlsx→parquet）→ schema（两阶段元数据）→ agent（两阶段 ReAct）→ sandbox（三层防护执行）**。

```
问题 → LLM 选表（失败降级启发式）→ 注入所选表 schema 画像
     → ReAct 循环：LLM 生成 pandas 代码 → AST 校验 → 子进程沙箱执行
                                              ↑ traceback 回填（自愈）
```

## 5.1 loader.py — 数据预转换

xlsx 解析慢（openpyxl），**启动时一次预转 parquet**（幂等：parquet 比 xlsx 新则跳过）：

```python
# loader.py 核心逻辑
if dst.exists() and not force and dst.stat().st_mtime >= src.stat().st_mtime:
    continue          # 已有且较新，跳过
df = pd.read_excel(src)
df.to_parquet(dst, index=False)
```

实测数据：`enterprise_promise` 26 万行 × 52 列解析约 100 秒——只发生一次，后续启动 `converted_now: 0`。

## 5.2 schema.py — 两阶段元数据

结构说明 docx 是无表边界的大字段表（约 2.3 万字符）。全量注入 system prompt 曾导致 7B 模型推理超时。方案：

1. **一次性解析**为全局"列名→中文描述"字典（进程内缓存 `_DOCS_CACHE`）
2. 代理先做表选择，再**只为所选表**生成紧凑画像（列名=描述 + dtype + 1 行示例）

`docx_to_markdown()` 同时是通用工具：段落按 Heading 层级转 md 标题，表格转 md 表格。

## 5.3 agent.py — 两阶段 ReAct 代理

### 阶段一：表选择（_select_tables）

LLM 输出 JSON `{"tables": [...]}`（temperature=0, max_tokens=200）；失败降级为 `_heuristic_select`——问题与表介绍的 **2-gram 重合度排序**，并强制补 `base_t_enterprise_draft`（企业有效性过滤常需要）。**永不抛异常**。

**结构隔离（2026-09-03 引入）**：选表消息拆为 system（选表指令 + 表清单）与 user（仅问题）两条，用户输入不再与指令同级拼接——这是输入侧注入防护的一部分（详见 [06 章](06-编排与API层.md)）。

### 阶段二：ReAct 循环（run）

**为什么文本协议而非 function calling**：实测硅基流动 Qwen2.5-7B 的 `tool_calls` 输出不稳定（伪 JSON 乱码、无法解析、单次 26s+）；`Thought/Action/Action Input` 三行协议 + 正则解析完全可控。

```python
# agent.py 协议解析（容忍代码围栏、多种结束形态）
_ACTION_RE = re.compile(
    r"Action:\s*(execute_pandas|finish)\s*\n\s*Action Input:\s*\n?(.+?)"
    r"(?=\n[ \t]*(?:Thought:|Action:|Observation:)\s|\Z)",
    re.DOTALL | re.IGNORECASE,
)
```

**业务规则写进 system prompt**（这是把领域知识固化进系统的关键）：

1. 企业编码查询必须过滤有效企业：`delete_status=0 AND check_status=2`
2. 主表含删除/审核状态的同样只统计有效数据
3. 特殊作业"计划+实施"双时间窗（`start_time~end_time` 与 `zysskssj~zyssjssj`）都要检查
4. 禁止多表合并成一张表处理，跨表用 merge/join

**自愈闭环**：沙箱执行失败的 traceback 原样回填下一轮 `Observation:`，LLM 读错误改代码再试，最多 8 轮。最后一轮强制 finish（不再执行代码——结果无处使用）。

**观测留存（2026-09-06 引入）**：`TabularAnswer.observations` 保存每轮沙箱 Observation（finish/未收敛时截取尾部 3 轮）——早期轮多为报错自愈，对数值溯源价值低。用途：经 `QAResult.evidence` 透传给评测侧事实一致性校验（[07 章](07-评测与可观测.md) 口径 v2），让"答案中的数字是否真由代码执行得出"可核对。

格式错误有专门的纠错反馈（`_FORMAT_ERROR`），模型重新输出协议。

## 5.4 sandbox.py — 三层防护（项目核心差异点）

**威胁模型**：LLM 生成的代码可能被提示注入操纵，目标任意文件读写/命令执行/secret 读取/数据外传。

**演进史（重要）**：最初黑名单方案被实测绕过——`np.memmap` 任意文件读、`pd.io.common.os` 赋值别名链读 secret + `execve` 命令执行——**黑名单在原理上无法穷尽**，改为能力白名单模型。

### 第一层：AST 静态校验（validate_code）

- 禁止 `import / global / nonlocal`
- 禁止一切下划线属性（阻断 `__class__`/`__globals__`/`__import__` 逃逸链）
- 属性黑名单 60+ 项：进程执行族（`execve/execv/spawnl...`）、pandas/numpy 文件读写原语（`read_csv/memmap/fromfile/to_csv...`）、网络（`urlopen/socket/httpx...`）
- **连续属性链深 ≤ 2**（挡 `pd.io.common.os.environ` 直链）
- **名字白名单**：仅 `pd/np/dfs` + 纯函数内建 + 异常类 + 已赋值变量

### 第二层：运行时白名单命名空间（子进程 globals）

子进程中的 `pd`/`np` **不是模块本体**，是 curated `SimpleNamespace`（仅 DataFrame/merge/to_datetime、array/mean/linalg 等分析函数）——`pd.io`、`np.memmap`、`np.core` **物理不存在**，赋值别名绕过（`c = pd.io.common; c.os`）必然 `AttributeError`。`__builtins__` 同为白名单字典。

> 即使 AST 层被完全绕过，可达能力面仍是封闭集合——这是纵深防御的含义。

### 第三层：进程隔离

独立子进程（`multiprocessing`，Windows spawn）执行、超时强杀（默认 30s `terminate()`——**Python 线程无法安全强杀（GIL），子进程是唯一确定性隔离边界**）、输出截断、表数据惰性加载（只 read 所需 parquet，避免大对象序列化）。

### 验证与已知边界

- `tests/test_sandbox.py`：**30 条攻击载荷**（动态导入/dunder 逃逸/numpy 文件原语/赋值别名链/进程执行族）静态或运行时全部拦截；合法分析（`np.linalg.norm`、`.str.contains`、`try/except`）正常执行；死循环被超时终止
- **已知边界（如实声明）**：子进程以当前 OS 用户权限运行；对抗国家级攻击者需 OS 级降权（seccomp/AppArmor/容器非 root）。当前威胁模型（提示注入驱动的代码逃逸）下三层已封闭已知攻击面。

下一章 → [06-编排与API层](06-编排与API层.md)
