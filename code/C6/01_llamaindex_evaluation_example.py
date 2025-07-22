import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import pandas as pd
import asyncio
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceWindowNodeParser, SentenceSplitter
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    RetrieverEvaluator,
    BatchEvalRunner,
)
from llama_index.core.evaluation.eval_utils import get_results_df
from llama_index.core.evaluation import DatasetGenerator, QueryResponseDataset

Settings.llm = DeepSeek(model="deepseek-chat", temperature=0.1, api_key=os.getenv("DEEPSEEK_API_KEY"))
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

async def main():
    # 1. 加载或生成评估数据集
    eval_dataset_path = "./c6_eval_dataset.json"
    if os.path.exists(eval_dataset_path):
        print(f"正在从 {eval_dataset_path} 加载已有的评估数据集...")
        eval_dataset = QueryResponseDataset.from_json(eval_dataset_path)
        print("评估数据集加载完毕。")
    else:
        print(f"未找到评估数据集，正在生成新的数据集并保存到 {eval_dataset_path}...")
        # 为了首次生成的速度，我们仅使用部分文档
        reader = SimpleDirectoryReader(input_files=["../../data/C3/pdf/IPCC_AR6_WGII_Chapter03.pdf"])
        documents = reader.load_data()
        dataset_generator = DatasetGenerator.from_documents(documents[:20])
        eval_dataset = await dataset_generator.agenerate_dataset_from_nodes(num=50)
        print("评估数据集生成完毕，正在保存...")
        eval_dataset.save_json(eval_dataset_path)
        print("保存成功。")

    # 提取文档用于后续索引构建
    reader = SimpleDirectoryReader(input_files=["../../data/C3/pdf/IPCC_AR6_WGII_Chapter03.pdf"])
    documents = reader.load_data()

    # 2. 构建两种不同的RAG查询引擎和检索器进行对比
    # 2.1 句子窗口检索
    sentence_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    sentence_nodes = sentence_parser.get_nodes_from_documents(documents)
    sentence_index = VectorStoreIndex(sentence_nodes)

    sentence_query_engine = sentence_index.as_query_engine(
        similarity_top_k=2,
        node_postprocessors=[
            MetadataReplacementPostProcessor(target_metadata_key="window")
        ],
    )
    sentence_retriever = sentence_index.as_retriever(similarity_top_k=2)

    # 2.2 常规分块检索（基准）
    base_parser = SentenceSplitter(chunk_size=512)
    base_nodes = base_parser.get_nodes_from_documents(documents)
    base_index = VectorStoreIndex(base_nodes)

    base_query_engine = base_index.as_query_engine(similarity_top_k=2)
    base_retriever = base_index.as_retriever(similarity_top_k=2)

    # 3. 初始化评估器
    # 响应评估器
    faithfulness_evaluator = FaithfulnessEvaluator(llm=Settings.llm)
    relevancy_evaluator = RelevancyEvaluator(llm=Settings.llm)

    # 4. 执行对比评估
    print("开始执行对比评估...")

    # 4.1 对句子窗口检索进行评估
    print("\n=== 句子窗口检索评估 ===")
    sentence_evaluators = {
        "faithfulness": faithfulness_evaluator,
        "relevancy": relevancy_evaluator,
    }
    sentence_runner = BatchEvalRunner(sentence_evaluators, workers=4, show_progress=True)

    # 提取问题列表
    queries = eval_dataset.queries

    sentence_eval_results = await sentence_runner.aevaluate_queries(
        queries=queries,
        query_engine=sentence_query_engine
    )

    # 4.2 对常规分块检索进行评估
    print("\n=== 常规分块检索评估 ===")
    base_evaluators = {
        "faithfulness": faithfulness_evaluator,
        "relevancy": relevancy_evaluator,
    }
    base_runner = BatchEvalRunner(base_evaluators, workers=4, show_progress=True)

    base_eval_results = await base_runner.aevaluate_queries(
        queries=queries,
        query_engine=base_query_engine
    )
    print("响应评估完成。")

    # 5. 分析并打印对比结果
    print("\n" + "="*80)
    print("评估结果对比分析")
    print("="*80)

    # 5.1 句子窗口检索结果
    if sentence_eval_results:
        sentence_faithfulness = sentence_eval_results.get("faithfulness")
        sentence_relevancy = sentence_eval_results.get("relevancy")

        if sentence_faithfulness and sentence_relevancy:
            sentence_faith_score = sum(r.passing for r in sentence_faithfulness) / len(sentence_faithfulness)
            sentence_rel_score = sum(r.passing for r in sentence_relevancy) / len(sentence_relevancy)

            print(f"\n--- 句子窗口检索评估结果 ---")
            print(f"忠实度 (Faithfulness): {sentence_faith_score:.2%}")
            print(f"相关性 (Relevancy): {sentence_rel_score:.2%}")

    # 5.2 常规分块检索结果
    if base_eval_results:
        base_faithfulness = base_eval_results.get("faithfulness")
        base_relevancy = base_eval_results.get("relevancy")

        if base_faithfulness and base_relevancy:
            base_faith_score = sum(r.passing for r in base_faithfulness) / len(base_faithfulness)
            base_rel_score = sum(r.passing for r in base_relevancy) / len(base_relevancy)

            print(f"\n--- 常规分块检索评估结果 ---")
            print(f"忠实度 (Faithfulness): {base_faith_score:.2%}")
            print(f"相关性 (Relevancy): {base_rel_score:.2%}")



if __name__ == "__main__":
    asyncio.run(main())
