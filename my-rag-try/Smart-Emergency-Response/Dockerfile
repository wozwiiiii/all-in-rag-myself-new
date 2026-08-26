# syntax=docker/dockerfile:1
# 多阶段构建：依赖层与代码层分离，充分利用层缓存
FROM python:3.12-slim AS builder
RUN pip install --no-cache-dir --prefix=/install \
    "fastapi>=0.110" \
    "uvicorn[standard]>=0.29" \
    "pydantic>=2.6" \
    "pydantic-settings>=2.2" \
    "chromadb>=0.5" \
    "openai>=1.30" \
    "tenacity>=8.2" \
    "pandas>=2.2" \
    "openpyxl>=3.1" \
    "pyarrow>=15.0" \
    "python-docx>=1.1" \
    "pypdf>=4.0" \
    "numpy>=1.26"

FROM python:3.12-slim
# 非 root 运行
RUN useradd --create-home appuser
WORKDIR /app
COPY --from=builder /install /usr/local
COPY app ./app
COPY scripts ./scripts
COPY eval ./eval
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=3)"
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
