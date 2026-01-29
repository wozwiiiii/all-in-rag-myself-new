# PowerRAG SDK 文本问答检索 Demo

本 Demo 用 PowerRAG（RAGFlow）Python SDK 实现最小闭环：上传一份 Markdown → 服务端解析分块并向量化 → 输入问题检索 top-k chunks。验收以 **检索出来的 chunks 是否语义相关** 为主，不要求 LLM 生成最终回答。

本目录遵循仓库 `code/` 的组织方式：提供 `main.py`、`config.py`、`.env.example`、`requirements.txt` 与可复现的样例数据。

---

## 1. 你会得到什么

- 一个可运行的脚本：`main.py`
- 一份样例文档：`data/sample.md`
- 一组示例问题：`data/questions.txt`
- 一套可复现的配置方式：`.env.example`

---

## 2. 前置条件

### 2.1 PowerRAG 服务已启动

你给的 docker-compose 环境变量里：

- `SVR_HTTP_PORT=9380` → PowerRAG/RAGFlow 服务对外 HTTP 端口通常是 `http://127.0.0.1:9380`
- `SVR_WEB_HTTP_PORT=80` → Web UI 走 `http://127.0.0.1`

先确认服务健康（至少应返回 200）：

```bash
curl -sS http://127.0.0.1:9380/v1/system/healthz
```

### 2.2 tenant 已配置默认 embedding（否则解析会 FAIL）

文档解析与检索依赖 embedding。若 tenant 的 `embd_id` 为空，你会看到类似错误：

- `Model(@None) not authorized`
- `Parse results: ... FAIL ...`

你可以用 UI 配，也可以用 API 配（推荐给无 UI / 服务器环境）。

下面给出一套通用的 API 配置流程：把一个可用的 embedding 模型绑定到当前 tenant，然后把 tenant 默认 `embd_id` 指向它。

---

## 3. 用 API 配好 embedding（通用）

这一步需要一个 Web 层的 `AUTH`（`/v1/*` 使用），它和 SDK 的 `ragflow-...` key 不是一回事。

你可以把 embedding 配置写进 `.env`（见 `.env.example` 的 `EMB_*`），下面命令会读取 `EMB_FACTORY/EMB_MODEL/EMB_API_BASE/EMB_API_KEY`。

### 3.1 获取 `AUTH`（注册并从响应头拿 Authorization）

PowerRAG 的 `/v1/user/register` 要求 password 先用服务端的 RSA public key 加密。最省事的方式是在容器内调用它自带的加密函数：

```bash
BASE_URL="http://127.0.0.1:9380"

ENC_PW="$(docker exec powerrag-powerrag-1 sh -lc 'python - <<\"PY\"\nfrom api.utils.crypt import crypt\nprint(crypt(\"powerrag\"))\nPY')"

EMAIL="powerrag.demo.$(date +%s)@example.com"
AUTH="$(curl -sS -D - -o /dev/null -X POST \"$BASE_URL/v1/user/register\" \\\n  -H 'Content-Type: application/json' \\\n  -d \"{\\\"nickname\\\":\\\"demo\\\",\\\"email\\\":\\\"$EMAIL\\\",\\\"password\\\":\\\"$ENC_PW\\\"}\" \\\n| awk 'BEGIN{IGNORECASE=1} /^authorization:/{print $2}' | tr -d '\\r')"
```

### 3.2 绑定 embedding 的外部 API

> 注意：`max_tokens` 需要显式传，否则可能报数据库字段错误。

```bash
curl -sS -X POST "$BASE_URL/v1/llm/add_llm" \
  -H "Authorization: $AUTH" \
  -H 'Content-Type: application/json' \
  -d '{
    "llm_factory": "'"${EMB_FACTORY}"'",
    "model_type": "embedding",
    "llm_name": "'"${EMB_MODEL}"'",
    "api_base": "'"${EMB_API_BASE}"'",
    "api_key": "'"${EMB_API_KEY}"'",
    "max_tokens": 8192
  }'
```

### 3.3 设置 tenant 默认 `embd_id`

```bash
TENANT_ID="$(curl -sS -H "Authorization: $AUTH" "$BASE_URL/v1/user/tenant_info" | python -c 'import sys,json; print(json.load(sys.stdin)["data"]["tenant_id"])')"

curl -sS -X POST "$BASE_URL/v1/user/set_tenant_info" \
  -H "Authorization: $AUTH" \
  -H 'Content-Type: application/json' \
  -d "{\"tenant_id\":\"$TENANT_ID\",\"llm_id\":\"\",\"embd_id\":\"${EMB_MODEL}@${EMB_FACTORY}\",\"asr_id\":\"\",\"img2txt_id\":\"\"}"
```

---

## 4. 生成 SDK 的 `ragflow-...` api_key

SDK 接口在 `/api/v1/*`，它不认 `AUTH`，需要 `ragflow-...` 这种 token（放在 header：`Authorization: Bearer <ragflow-...>`）。

用 `AUTH` 创建一个 SDK key：

```bash
DIALOG_ID="$(python -c 'import uuid; print(uuid.uuid4().hex)')"
API_KEY="$(curl -sS -X POST "$BASE_URL/v1/api/new_token" \
  -H "Authorization: $AUTH" \
  -H 'Content-Type: application/json' \
  -d "{\"dialog_id\":\"$DIALOG_ID\"}" \
| python -c 'import sys,json; print(json.load(sys.stdin)["data"]["token"])')"

echo "$API_KEY"
```

---

## 5. 安装依赖与运行

### 5.1 安装依赖

```bash
cd code/powerrag-text-qa-demo
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 5.2 配置环境变量

复制一份 `.env`：

```bash
cp .env.example .env
```

填入：

- `RAGFLOW_BASE_URL=http://127.0.0.1:9380`
- `RAGFLOW_API_KEY=<上一步生成的 ragflow-...>`

### 5.3 运行 demo

```bash
python main.py \
  --file data/sample.md \
  --question "已发货未签收的退款规则是什么？" \
  --top-k 5 \
  --cleanup
```

你会看到两段输出：

1) `Parse results`：文档解析与分块状态（期望 `DONE`）  
2) `Retrieved chunks`：top-k chunks 的内容预览与相似度字段（若 SDK 返回）  

---

## 6. 实现思路（对应 issue 验收点）

1) 初始化 SDK：`ragflow_sdk.RAGFlow(api_key, base_url)`  
2) 上传文件：`dataset.upload_documents([{display_name, blob}])`  
3) 解析分块：`dataset.parse_documents([doc.id])`（同步等待完成，便于 demo 验收）  
4) 检索：`rag.retrieve(question=..., dataset_ids=[...], document_ids=[...], page_size=top_k)`  

---

## 7. Troubleshooting

### 7.1 Parse results 是 FAIL

优先检查 tenant 的 `embd_id` 是否为空、是否指向一个已配置且可用的 embedding（见第 3 节）。

如果已经配置仍失败，直接看 task executor 日志最省时间：

```bash
docker exec powerrag-powerrag-1 sh -lc 'tail -n 200 /ragflow/logs/task_executor_* | tail -n 200'
```

如果你看到类似 `OBConnection.createIndex error` / `Column object 'id' already assigned ...` 这类 OceanBase 文档引擎的建表/反射报错，一般不是 demo 代码问题，需要从 PowerRAG 侧排查（必要时重启服务或复用同一个 tenant 进行实验）。
