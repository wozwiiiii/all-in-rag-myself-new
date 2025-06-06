import os
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from modelscope.hub.snapshot_download import snapshot_download

load_dotenv()

# 模型路径

model_dir = snapshot_download(
    'BAAI/bge-small-zh-v1.5',
    allow_patterns=[
        'config.json',
        'model.safetensors',
        'modules.json',
        'sentence_bert_config.json',
        'special_tokens_map.json',
        'tokenizer.json',
        'tokenizer_config.json',
        'vocab.txt',
        '1_Pooling/*'
    ],
    local_dir='models/bge-small-zh-v1.5'
)

markdown_path = "data/C1/markdown/easy-rl-chapter1.md"

# 加载本地markdown文件
loader = UnstructuredMarkdownLoader(markdown_path)
docs = loader.load()

# 文档分块
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(docs)

# 中文嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name=model_dir,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
  
# 创建向量存储
vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(texts)

# 构建用户查询
question = "强化学习与监督学习有什么区别？"

# 在向量存储中搜索相关文档，并准备上下文内容
retrieved_docs = vectorstore.similarity_search(question, k=3)
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

# 提示词模板
prompt = ChatPromptTemplate.from_template("""
                基于以下上下文，回答问题。如果上下文中没有相关信息，
                请说"我无法从提供的上下文中找到相关信息"。
                上下文: {context}
                问题: {question}
                回答:"""
                                          )

# 调用大语言模型生成答案
llm = ChatDeepSeek(
    model="deepseek-chat",  # DeepSeek API 支持的模型名称
    temperature=0.7,        # 控制输出的随机性
    max_tokens=2048,        # 最大输出长度
    api_key=os.getenv("DEEPSEEK_API_KEY")  # 从环境变量加载API key
)

answer = llm.invoke(prompt.format(question=question, context=docs_content))
print(answer)

