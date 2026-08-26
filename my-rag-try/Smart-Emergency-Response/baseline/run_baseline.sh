#!/bin/bash

echo "开始执行脚本..."

cd baseline/
echo "已进入 baseline/ 目录"

export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
export OPENAI_API_BASE="https://api.siliconflow.cn/v1"
export OPENAI_API_MODEL="Qwen/Qwen2.5-7B-Instruct"
export OPENAI_API_EMBEDDING_MODEL="BAAI/bge-m3"
echo "环境变量设置完成"

pip install llama-index chromadb openai python-dotenv llama-index-vector-stores-chroma docx2txt notebook "numpy<2.0"
echo "必要的Python包安装完成"

pip install --upgrade pip
pip install --upgrade --force-reinstall numpy scipy
echo "pip和科学计算库更新完成"

python main.py
echo "main.py执行完成"