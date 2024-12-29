import streamlit as st
import data_utils


def init_page():
    """
    初始化页面配置，设置标题和布局等。
    """
    st.set_page_config(page_title="📄 基于LangChain和DeepSeek的本地文档问答系统", layout="wide")
    st.title("📄 基于LangChain和DeepSeek的本地文档问答系统")


def render_document_upload_sidebar():
    """
    渲染左侧栏的文件上传控件，并返回上传的文件对象列表。
    """
    st.sidebar.header("上传文件: PDF、 Word、 Markdown、 txt")
    uploaded_files = st.sidebar.file_uploader(
        "选择文件",
        type=["pdf", "doc", "docx", "md", "txt"],
        accept_multiple_files=True
    )
    return uploaded_files


def render_session_selection_sidebar():
    """
    侧边栏会话管理：包含创建、选择、删除会话功能。
    当创建新会话后，会立刻切换到新会话。
    """
    st.sidebar.header("会话管理")

    # 1) 获取已有会话列表
    sessions = data_utils.get_sessions()

    # 2) 输入新会话名称
    current_input = st.session_state.get("new_session", "")
    exists = (current_input in sessions) if current_input else False
    placeholder = "该名称已存在" if exists else "请输入新对话名称"

    new_session = st.sidebar.text_input(
        "创建新会话",
        key="new_session",
        placeholder=placeholder,
    )

    # 3) 如果输入了新会话名称且名称不重复，则创建并跳转
    if new_session and not exists:
        if st.sidebar.button("确定创建新会话"):
            sessions.append(new_session)
            data_utils.save_sessions(sessions)
            st.session_state.selected_session = new_session
            # 这里立刻跳转到新会话
            st.rerun()

    # 4) 下拉框选择已有会话
    selected_session = None
    if sessions:
        selected_session = st.sidebar.selectbox("选择会话", sessions,
                                                index=sessions.index(
                                                    st.session_state.get("selected_session", sessions[0]))
                                                if st.session_state.get("selected_session") else 0
                                                )

    # 5) 删除当前选中会话
    if selected_session:
        if st.sidebar.button("删除当前会话", key="delete_session"):
            data_utils.delete_session(selected_session)
            updated_sessions = data_utils.get_sessions()
            if updated_sessions:
                # 如果还有其他会话，就选第一个
                st.session_state.selected_session = updated_sessions[0]
            else:
                st.session_state.selected_session = None
            st.rerun()

    if selected_session != st.session_state.get("selected_session"):
        # 清除相关的会话状态
        if 'qa_chain' in st.session_state:
            st.session_state.qa_chain = {}
        st.session_state.selected_session = selected_session
        st.rerun()

    return selected_session


def display_chat_history(messages):
    """
    显示一组对话消息（用户、助手或源文档）。
    通过多列布局来实现左右分栏：用户消息在右，助手消息在左。
    源文档通过助手消息下方的折叠面板显示。
    """
    for role, content in messages:
        if role == "user":
            # 用户消息：右侧
            _, col_main, col_user = st.columns([0.2, 0.7, 0.1])
            with col_main:
                st.info(content)
            with col_user:
                st.markdown("**用户**")
        elif role == "assistant":
            # 助手消息：左侧
            col_assistant, col_main, _ = st.columns([0.1, 0.7, 0.2])
            with col_assistant:
                st.markdown(f"**助手**")
            with col_main:
                st.success(content)



def get_user_question():
    """
    Streamlit 的 chat_input，用于获取用户输入问题。
    """
    return st.chat_input("请输入您的问题：", key="user_input")


def display_loading_message(stop_key):
    """
    显示“正在回答中...”和“❌ 终止回答”按钮，并返回:
      - placeholder: 容器，用于后续 .empty() 一次性清空
      - stop_clicked: 按钮是否被点击
    """
    placeholder = st.empty()  # 创建一个空占位
    with placeholder.container():
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            st.markdown("**正在回答中...**")
        with col2:
            stop_clicked = st.button("❌ 终止回答", key=stop_key)

    return placeholder, stop_clicked
