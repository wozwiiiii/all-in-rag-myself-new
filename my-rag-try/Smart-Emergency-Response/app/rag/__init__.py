"""文本 RAG 子系统：分块（chunking）→ 索引（ingest）→ 融合检索（retriever）
→ 生成（generator），稀疏路（bm25）为可降级组件。

注意：chromadb 为运行时懒加载依赖——缺失时模块可导入，检索器构造时才报错。
"""
