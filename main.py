import os
import tempfile

import streamlit as st
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

import data_utils  # 负责文件处理、向量数据库构建等工具函数
import ui  # 负责页面UI相关的函数（这里已修改为多栏对话）

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
        # 初始化一个全局的 vector_store
        st.session_state.vector_store = data_utils.initialize_vector_store()

    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = {}

    if 'fallback_chain' not in st.session_state:
        st.session_state.fallback_chain = fallback_chain

    st.session_state.vector_store_path = data_utils.VECTOR_STORE_PATH

    # 加载对话历史
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = {}

    if 'selected_session' not in st.session_state:
        st.session_state.selected_session = None
    if selected_session not in st.session_state.conversation_history:
        st.session_state.conversation_history[selected_session] = data_utils.initialize_conversation_history(
            selected_session)
    # 左侧上传PDF文件（支持多文件）
    uploaded_files = ui.render_document_upload_sidebar()

    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            try:
                # 获取文件扩展名
                _, ext = os.path.splitext(uploaded_file.name)
                ext = ext.lower()

                if ext not in ['.pdf', '.docx', '.txt', '.md']:
                    st.sidebar.error(f"不支持的文件类型: {ext}")
                    continue

                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name

                text = data_utils.load_document(tmp_file_path)
                if text:
                    doc_chunks = data_utils.prepare_documents(text)
                    documents.extend(doc_chunks)
                    st.sidebar.success(f"已成功加载 {uploaded_file.name}")

                # 清理临时文件
                os.unlink(tmp_file_path)

            except Exception as e:
                st.sidebar.error(f"处理文件 {uploaded_file.name} 时出错: {e}")
        if documents:
            with st.spinner("正在构建或更新向量数据库，请稍候..."):
                if st.session_state.vector_store is None:
                    # 如果 vector_store 未初始化，则创建新的向量存储
                    st.session_state.vector_store = data_utils.build_vector_store_from_documents(
                        documents, st.session_state.vector_store_path)
                else:
                    # 否则，添加文档到现有的向量存储
                    data_utils.add_documents_to_vector_store(
                        st.session_state.vector_store,
                        documents
                    )

    # 如果已经有 vector_store 并且 qa_chain 未初始化
    if st.session_state.vector_store and selected_session not in st.session_state.qa_chain:
        retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 5, "score_threshold": 0.7})
        st.session_state.qa_chain[selected_session] = ConversationalRetrievalChain.from_llm(
            llm=chat_model,
            retriever=retriever,
            return_source_documents=True,
            verbose=True
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
        if st.session_state.qa_chain.get(selected_session) and st.session_state.vector_store:
            # 尝试从文档中回答
            response = data_utils.conversational_answer(
                chain=st.session_state.qa_chain[selected_session],
                question=user_question,
                stop_flag=stop_flag,
                history=st.session_state.conversation_history[selected_session],

            )
            answer = response.get("answer", "抱歉，我无法生成回答。")
            source_documents = response.get("source_documents", [])

            # 如果回答为空或未找到回答，则使用 fallback_chain
            if not source_documents:
                fallback_answer = data_utils.conversational_answer(
                    chain=st.session_state.fallback_chain,
                    question=user_question,
                    stop_flag=stop_flag,
                    history=st.session_state.conversation_history[selected_session]
                )
                st.session_state.conversation_history[selected_session].append(
                    ("assistant", "在文档中未找到相关信息，以下是基于大模型的回答：" + fallback_answer.get('answer')))
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
            st.session_state.conversation_history[selected_session].append(
                ("assistant", "vector_store未创建" + fallback_answer.get('answer')))

        # 保存对话历史
        data_utils.save_conversation_history(selected_session, st.session_state.conversation_history[selected_session])

        # 清空“正在回答中...”提示
        loading_message.empty()

        # 显示更新后的对话历史
        ui.display_chat_history(st.session_state.conversation_history[selected_session])


if __name__ == "__main__":
    main()