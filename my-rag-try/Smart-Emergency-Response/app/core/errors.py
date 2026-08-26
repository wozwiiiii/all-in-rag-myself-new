"""统一业务异常体系：API 层据此生成统一 JSON 错误体，不向外泄漏堆栈。"""


class AppError(Exception):
    code: str = "internal_error"
    http_status: int = 500

    def __init__(self, message: str, *, code: str | None = None, http_status: int | None = None):
        super().__init__(message)
        self.message = message
        if code:
            self.code = code
        if http_status:
            self.http_status = http_status

    def to_dict(self) -> dict:
        return {"error": {"code": self.code, "message": self.message}}


class ConfigError(AppError):
    code = "config_error"
    http_status = 500


class AuthError(AppError):
    code = "unauthorized"
    http_status = 401


class RateLimitError(AppError):
    code = "rate_limited"
    http_status = 429


class LLMError(AppError):
    code = "llm_unavailable"
    http_status = 502


class SandboxViolation(AppError):
    """LLM 生成的代码未通过 AST 白名单校验。"""

    code = "sandbox_violation"
    http_status = 400


class SandboxTimeout(AppError):
    code = "sandbox_timeout"
    http_status = 504


class RetrievalUnavailable(AppError):
    code = "retrieval_unavailable"
    http_status = 503
