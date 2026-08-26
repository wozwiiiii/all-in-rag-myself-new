"""API 认证与限流：X-API-Key 多 key 校验 + 每 key 滑动窗口限流。

已知限制（README 中声明）：限流状态在进程内存中，单实例有效；
多实例部署需替换为 Redis 等共享存储实现。
"""

from __future__ import annotations

import hmac
import threading
import time
from collections import defaultdict, deque

from fastapi import Request

from app.core.config import Settings
from app.core.errors import AuthError, RateLimitError

WINDOW_SECONDS = 60.0


class RateLimiter:
    def __init__(self, limit_per_minute: int):
        self._limit = limit_per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and q[0] <= now - WINDOW_SECONDS:
                q.popleft()
            if len(q) >= self._limit:
                retry_after = int(WINDOW_SECONDS - (now - q[0])) + 1
                err = RateLimitError(
                    f"请求过于频繁，每分钟限 {self._limit} 次，{retry_after}s 后重试"
                )
                err.retry_after = retry_after  # type: ignore[attr-defined] # handler 转为 Retry-After 头
                raise err
            q.append(now)


def make_api_key_dependency(settings: Settings, limiter: RateLimiter):
    async def verify(request: Request) -> str:
        provided = request.headers.get("X-API-Key", "")
        if settings.auth_enabled:
            # 常数时间比较，防时序侧信道逐字节探测
            if not any(hmac.compare_digest(provided, k) for k in settings.api_key_list):
                raise AuthError("无效的 API Key（请通过 X-API-Key 请求头提供）")
            identity = provided
        else:
            # 认证关闭（仅限本地开发）：按客户端 IP 限流，避免匿名共享桶互相 DoS
            identity = f"ip:{request.client.host if request.client else 'unknown'}"
        limiter.check(identity)
        return identity

    return verify
