import os
import json
import time
import tempfile
import streamlit as st

from pdfminer.high_level import extract_text
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  # 请确保安全管理

VECTOR_STORE_DIR = "vector_stores"
CONVERSATION_HISTORY_DIR = "conversation_histories"

def load_text_file(file_path):
    """从文本文件中加载全文内容"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def load_pdf(file_path):
    """从PDF加载全文文本"""
    try:
        text = extract_text(file_path)
        if not text.strip():
            st.warning(f"从PDF文件 {file_path} 中未提取到任何文本。")
        else:
            st.write(f"成功从PDF文件 {file_path} 中提取了 {len(text)} 个字符。")
        return text
    except Exception as e:
        st.error(f"从PDF提取文本时出错: {e}")
        return ""

def load_document(file_path):
    """根据文件扩展名加载不同类型的文档"""
    _, ext = os.path.splitext(file_path)
    if ext.lower() == '.pdf':
        return load_pdf(file_path)
    else:
        return load_text_file(file_path)

def prepare_documents(text, chunk_size=1024, chunk_overlap=128):
    """
    将文本拆分为多个文档块，并转换为 Document 对象列表。
    """
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    st.write(f"生成了 {len(chunks)} 个文本块。")

    if not chunks:
        st.error("文本分块后未生成任何文本块。请检查PDF内容和分块参数。")
        return []

    documents = [Document(page_content=chunk) for chunk in chunks]
    return documents

def build_vector_store_from_documents(documents, vector_store_path):
    """
    从文档列表构建向量数据库，并保存到指定路径。
    返回一个 FAISS vector_store 对象。
    """
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
    vector_store = FAISS.from_documents(documents, embeddings)

    # 使用 vector_store.index.ntotal 获取嵌入向量的数量
    st.write(f"生成了 {vector_store.index.ntotal} 个嵌入向量。")
    if vector_store.index.ntotal == 0:
        st.error("嵌入生成失败，未生成任何向量。请检查嵌入模型和文档内容。")
        return None

    vector_store.save_local(vector_store_path)
    st.success(f"向量数据库已保存到 {vector_store_path}")
    return vector_store

def add_documents_to_vector_store(vector_store, documents):
    """
    将新的文档添加到现有的向量数据库中。
    """
    vector_store.add_documents(documents)  # 仅传递 documents
    st.write(f"已添加 {len(documents)} 个文档到向量数据库。")

def initialize_conversation_history(session_id):
    """初始化对话历史，根据会话ID加载对应的JSON文件"""
    os.makedirs(CONVERSATION_HISTORY_DIR, exist_ok=True)
    history_file = os.path.join(CONVERSATION_HISTORY_DIR, f"{session_id}.json")
    if os.path.exists(history_file):
        if os.path.getsize(history_file) == 0:
            return []
        else:
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                    history = [tuple(pair) for pair in history]
                    return history
            except json.JSONDecodeError:
                st.warning("对话历史文件内容无效，将初始化为空。")
                return []
    else:
        return []

def save_conversation_history(session_id, history):
    """保存对话历史到对应的JSON文件"""
    os.makedirs(CONVERSATION_HISTORY_DIR, exist_ok=True)
    history_file = os.path.join(CONVERSATION_HISTORY_DIR, f"{session_id}.json")
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump([list(pair) for pair in history], f, ensure_ascii=False, indent=4)

def simulate_answer_generation(chain, question, stop_flag, history):
    """
    演示一个“分步”生成的过程，携带一个 stop_flag
    来模拟中途停止回答的情况。
    """
    for _ in range(3):
        time.sleep(1)
        if stop_flag["stop"]:
            return "回答已终止。"
    # 如果没有停止，则真正执行 chain.run
    return chain.run(question=question, chat_history=history)

def conversational_answer(chain, question, stop_flag, history):
    """通用的多轮对话回答流程，返回最终 answer"""
    try:
        response = simulate_answer_generation(chain, question, stop_flag, history)
        if not isinstance(response, str):
            st.error("返回的回答格式不正确。")
            return "抱歉，我无法生成回答。"
        if not response.strip():
            return "未找到回答。"
        return response
    except Exception as e:
        st.error(f"在生成回答时出错: {e}")
        return "抱歉，我无法生成回答。"
