"""API 请求/响应模型：入参长度约束在边界处完成，业务层不再重复校验。"""

from pydantic import BaseModel, Field


class QARequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=500, description="用户问题（2-500 字）")


class QASource(BaseModel):
    source: str
    heading: str | None = None
    score: float | None = None
    kind: str | None = None


class UsageMeta(BaseModel):
    """单次问答请求实际消耗的 token 用量（观测用途）。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0


class QAResponse(BaseModel):
    answer: str
    route: str
    sources: list[QASource] = []
    refused: bool = False
    latency_ms: int
    request_id: str
    usage: UsageMeta | None = None


class HealthResponse(BaseModel):
    status: str
    version: str


class ReadyResponse(BaseModel):
    status: str
    checks: dict[str, bool]
