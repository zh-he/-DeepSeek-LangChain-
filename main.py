import os
import tempfile
import streamlit as st

from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

import ui  # 负责页面UI相关的函数（这里已修改为多栏对话）
import data_utils  # 负责PDF/Text处理、向量数据库构建等工具函数

# 从环境变量读取 DeepSeek 的 API Key
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  # 请确保安全

# 使用 LangChain 的 ChatOpenAI 来对接 DeepSeek (OpenAI兼容)
chat_model = ChatOpenAI(
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base="https://api.deepseek.com",  # DeepSeek的base_url
    model_name="deepseek-chat",
    max_tokens=2048,
    temperature=0.7,
    top_p=0.9
)

# 初始化Prompt模板，用于直接使用LLM回答
fallback_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are a helpful assistant. Answer the following question based solely on your training data. 
Please don't make up any contents.

Question: {question}
Answer:
"""
)
fallback_chain = LLMChain(llm=chat_model, prompt=fallback_prompt)

def main():
    ui.init_page()  # 初始化页面

    # 侧边栏会话管理
    selected_session = ui.render_session_selection_sidebar()

    if not selected_session:
        st.warning("请创建或选择一个会话。")
        return

    # 初始化 session_state 变量
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = {}
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = {}
    if 'fallback_chain' not in st.session_state:
        st.session_state.fallback_chain = fallback_chain

    # 加载或初始化会话相关的数据
    if selected_session not in st.session_state.vector_store:
        # 定义 vector_store_path 并初始化为 None
        vector_store_path = os.path.join(data_utils.VECTOR_STORE_DIR, f"{selected_session}.faiss")
        st.session_state.vector_store[selected_session] = None
        st.session_state.vector_store_path = vector_store_path
    else:
        vector_store_path = os.path.join(data_utils.VECTOR_STORE_DIR, f"{selected_session}.faiss")
        st.session_state.vector_store_path = vector_store_path

    if selected_session not in st.session_state.qa_chain:
        st.session_state.qa_chain[selected_session] = None

    # 加载对话历史
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = {}
    if selected_session not in st.session_state.conversation_history:
        st.session_state.conversation_history[selected_session] = data_utils.initialize_conversation_history(selected_session)

    # 左侧上传PDF文件（支持多文件）
    uploaded_files = ui.render_pdf_upload_sidebar()

    # 处理文件上传
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            # 处理多文件：将每个文件都存到临时文件再解析
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            st.sidebar.info(f"正在处理文件: {uploaded_file.name}")

            # 从文件中加载文本
            text = data_utils.load_document(tmp_file_path)
            if text:
                # 将文档拆分成多个块
                doc_chunks = data_utils.prepare_documents(text)
                documents.extend(doc_chunks)
                st.sidebar.success(f"已成功加载 {uploaded_file.name}")
            else:
                st.sidebar.error(f"无法从 {uploaded_file.name} 中提取文本。")

        if documents:
            with st.spinner("正在构建或更新向量数据库，请稍候..."):
                if st.session_state.vector_store[selected_session] is None:
                    # 构建新的向量数据库
                    st.session_state.vector_store[selected_session] = data_utils.build_vector_store_from_documents(
                        documents,
                        vector_store_path
                    )
                else:
                    # 添加新的文档到现有的向量数据库
                    data_utils.add_documents_to_vector_store(
                        st.session_state.vector_store[selected_session],
                        documents
                    )
                st.session_state.vector_store[selected_session].save_local(vector_store_path)
                st.sidebar.success("向量数据库已更新。")

    # 如果已经有 vector_store 并且 qa_chain 未初始化
    if st.session_state.vector_store[selected_session] and st.session_state.qa_chain[selected_session] is None:
        retriever = st.session_state.vector_store[selected_session].as_retriever(search_kwargs={"k": 2})
        st.session_state.qa_chain[selected_session] = ConversationalRetrievalChain.from_llm(
            llm=chat_model,
            retriever=retriever,
            return_source_documents=False  # 设置为 False，以简洁输出
        )

    # 显示完整的对话历史（多栏布局）
    ui.display_chat_history(st.session_state.conversation_history[selected_session])

    # 获取用户输入
    user_question = ui.get_user_question()

    if user_question:
        # 记录用户的提问
        st.session_state.conversation_history[selected_session].append(("user", user_question))
        # 保存对话历史
        data_utils.save_conversation_history(selected_session, st.session_state.conversation_history[selected_session])

        # 先在对话区域显示用户消息（多栏显示）
        ui.display_chat_history([("user", user_question)])

        # 用于中止回答的标志
        stop_flag = {"stop": False}

        # 显示“正在回答中...”提示 & 中止按钮
        loading_message, stop = ui.display_loading_message(stop_key=f"stop_button_{selected_session}_{user_question}")
        if stop:
            stop_flag["stop"] = True

        # 生成回答
        if st.session_state.qa_chain[selected_session] and st.session_state.vector_store[selected_session]:
            # 尝试从文档中回答
            answer = data_utils.conversational_answer(
                chain=st.session_state.qa_chain[selected_session],
                question=user_question,
                stop_flag=stop_flag,
                history=st.session_state.conversation_history[selected_session]
            )
            # 如果回答为空或未找到回答，则使用 fallback_chain
            if not answer.strip() or answer == "未找到回答。":
                st.session_state.conversation_history[selected_session].append(
                    ("assistant", "在文档中未找到相关信息，以下是基于大模型的回答：")
                )

                fallback_answer = data_utils.conversational_answer(
                    chain=st.session_state.fallback_chain,
                    question=user_question,
                    stop_flag=stop_flag,
                    history=st.session_state.conversation_history[selected_session]
                )
                st.session_state.conversation_history[selected_session].append(("assistant", fallback_answer))
            else:
                st.session_state.conversation_history[selected_session].append(("assistant", answer))
        else:
            # 如果没有 vector_store 则直接使用 fallback_chain
            fallback_answer = data_utils.conversational_answer(
                chain=st.session_state.fallback_chain,
                question=user_question,
                stop_flag=stop_flag,
                history=st.session_state.conversation_history[selected_session]
            )
            st.session_state.conversation_history[selected_session].append(("assistant", fallback_answer))

        # 保存对话历史
        data_utils.save_conversation_history(selected_session, st.session_state.conversation_history[selected_session])

        # 清空“正在回答中...”提示
        loading_message.empty()

        # 显示更新后的对话历史
        ui.display_chat_history(st.session_state.conversation_history[selected_session])

    if 'fallback_chain' not in st.session_state:
        st.session_state.fallback_chain = fallback_chain

if __name__ == "__main__":
    main()
