import streamlit as st
import os
import data_utils
def init_page():
    """
    初始化页面配置，设置标题和布局等。
    """
    st.set_page_config(page_title="本地文档问答系统", layout="wide")
    st.title("📄 本地文档问答系统")

def render_pdf_upload_sidebar():
    """
    渲染左侧栏的PDF文件上传控件，并返回上传的文件对象列表。
    """
    st.sidebar.header("上传PDF文件")
    uploaded_files = st.sidebar.file_uploader(
        "选择PDF文件",
        type=["pdf"],
        accept_multiple_files=True
    )
    return uploaded_files

def render_session_selection_sidebar():
    """
    渲染左侧栏的会话选择控件，并返回选中的会话ID。
    """
    st.sidebar.header("会话管理")
    sessions = get_existing_sessions()

    # 初始化 'selected_session' 在 session_state 中
    if 'selected_session' not in st.session_state:
        if sessions:
            st.session_state.selected_session = sessions[0]
        else:
            st.session_state.selected_session = None

    # 如果会话列表为空，显示提示信息
    if not sessions:
        st.sidebar.info("暂无会话，请点击下方按钮创建新会话。")

    # 渲染会话选择下拉框
    selected_session = st.sidebar.selectbox(
        "选择会话",
        options=sessions,
        index=0 if sessions else -1,
        key='selected_session'
    )

    # 定义一个回调函数，用于新建会话
    def create_new_session_callback():
        new_session_id = create_new_session()
        # 创建新会话的历史文件
        data_utils.save_conversation_history(new_session_id, [])
        # 更新 session_state 以选中新会话
        st.session_state.selected_session = new_session_id
        # 重新运行应用以刷新会话列表并选中新的会话
        st.rerun()  # 如果使用最新版本的 Streamlit，可以使用 st.rerun()

    # 渲染“新建会话”按钮，并绑定回调函数
    st.sidebar.button("新建会话", on_click=create_new_session_callback, key="new_session_button")

    return st.session_state.selected_session



def get_existing_sessions():
    """
    获取现有会话的列表（基于 conversation_histories 目录中的 JSON 文件）。
    """
    history_dir = "conversation_histories"
    if not os.path.exists(history_dir):
        return []
    files = os.listdir(history_dir)
    sessions = [os.path.splitext(file)[0] for file in files if file.endswith('.json')]
    return sessions

def create_new_session():
    """
    创建一个新的会话ID，并返回该ID。
    """
    import uuid
    new_session_id = f"session_{uuid.uuid4().hex[:8]}"
    return new_session_id

def display_chat_history(messages):
    """
    显示一组对话消息（用户或助手）。
    通过多列布局 (st.columns) 来实现左右分栏：用户消息在左，助手消息在右。
    """
    for role, content in messages:
        if role == "user":
            # 用户消息：左侧
            col1, col2, _ = st.columns([0.1, 0.7, 0.2])
            with col1:
                st.markdown("**用户**")
            with col2:
                st.info(content)
        else:
            # 助手消息：右侧
            _, col2, col3 = st.columns([0.2, 0.7, 0.1])
            with col3:
                st.markdown("**助手**")
            with col2:
                st.success(content)

def get_user_question():
    """
    底部文本输入框，用于获取用户输入问题。
    """
    return st.chat_input("请输入您的问题：", key="user_input")

def display_loading_message(stop_key):
    """
    在回答生成过程中，显示“正在回答中...”以及可中止按钮。
    """
    with st.container():
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            loading_message = st.markdown("**正在回答中...**")
        with col2:
            stop = st.button("❌ 终止回答", key=stop_key)
        return loading_message, stop

def display_final_answer(answer):
    """
    生成最终回答后，在聊天区域显示助手消息（如果需要单独调用）。
    """
    _, col2, col3 = st.columns([0.2, 0.7, 0.1])
    with col3:
        st.markdown("**助手**")
    with col2:
        st.success(answer)
