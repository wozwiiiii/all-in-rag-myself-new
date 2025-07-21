import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import pandas as pd
import asyncio
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    RetrieverEvaluator,
    BatchEvalRunner,
)
from llama_index.core.evaluation.eval_utils import get_results_df
from llama_index.core.evaluation import DatasetGenerator

Settings.llm = DeepSeek(model="deepseek-chat", temperature=0.1, api_key=os.getenv("DEEPSEEK_API_KEY"))
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

async def main():
    # 1. 加载数据并生成评估集
    reader = SimpleDirectoryReader(input_files=["../../data/C3/pdf/IPCC_AR6_WGII_Chapter03.pdf"])
    documents = reader.load_data()
    
    print("正在从部分文档生成评估数据集(前10页)...")
    # 为了加速，仅使用前10页文档生成评估数据
    dataset_generator = DatasetGenerator.from_documents(documents[:10])
    eval_dataset = await dataset_generator.agenerate_dataset_from_nodes(num=30)
    print("评估数据集生成完毕。")

    # 2. 构建RAG查询引擎和检索器
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    nodes = node_parser.get_nodes_from_documents(documents)
    index = VectorStoreIndex(nodes)
    
    query_engine = index.as_query_engine(similarity_top_k=2)
    retriever = index.as_retriever(similarity_top_k=2)

    # 3. 初始化评估器
    # 响应评估器
    faithfulness_evaluator = FaithfulnessEvaluator(llm=Settings.llm)
    relevancy_evaluator = RelevancyEvaluator(llm=Settings.llm)
    
    # 检索评估器
    retriever_evaluator = RetrieverEvaluator.from_metric_names(
        ["mrr", "hit_rate"], retriever=retriever
    )

    # 4. 使用BatchEvalRunner执行批量评估
    print("开始执行批量评估...")
    # 定义评估器字典
    evaluators = {
        "faithfulness": faithfulness_evaluator,
        "relevancy": relevancy_evaluator,
        "retrieval": retriever_evaluator
    }
    
    # 创建批量评估运行器
    runner = BatchEvalRunner(evaluators, workers=4, show_progress=True)
    
    # 执行评估
    eval_results = await runner.aevaluate_dataset(eval_dataset, query_engine=query_engine)
    print("批量评估完成。")

    # 5. 分析并打印结果
    # 提取检索评估结果
    retrieval_eval_results = eval_results.get("retrieval")
    if retrieval_eval_results:
        retrieval_df = get_results_df([retrieval_eval_results])
        print("\n--- 检索评估结果 (Retrieval Evaluation) ---")
        print(retrieval_df)
    
    # 提取响应评估结果
    faithfulness_eval_results = eval_results.get("faithfulness")
    relevancy_eval_results = eval_results.get("relevancy")
    if faithfulness_eval_results and relevancy_eval_results:
        response_df_data = {
            "question": [item.query for item in eval_dataset.rag_questions],
            "response": [result.response for result in eval_results["relevancy"]],
            "faithfulness": [r.passing for r in faithfulness_eval_results],
            "relevancy": [r.passing for r in relevancy_eval_results],
        }
        response_df = pd.DataFrame(response_df_data)
        print("\n--- 响应评估结果 (Response Evaluation) ---")
        print(response_df.to_string())

if __name__ == "__main__":
    asyncio.run(main())
