
import streamlit as st
import uuid

from backend import chatbot, ingest_pdf
from langchain_core.messages import HumanMessage, AIMessage


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Nitin Made CHATBOT",
    page_icon="📄",
    layout="wide"
)


# =========================================================
# GENERATE THREAD ID
# =========================================================

def generate_thread_id():
    return str(uuid.uuid4())


# =========================================================
# ADD THREAD
# =========================================================

def add_thread(thread_id):

    if thread_id not in st.session_state["chat_thread"]:
        st.session_state["chat_thread"].append(thread_id)


# =========================================================
# RESET CHAT
# =========================================================

def reset_chat():

    thread_id = generate_thread_id()

    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []

    add_thread(thread_id)


# =========================================================
# LOAD CONVERSATION
# =========================================================

def load_conversation(thread_id):

    try:

        state = chatbot.get_state(
            config={
                "configurable": {
                    "thread_id": thread_id
                }
            }
        )

        return state.values.get("messages", [])

    except Exception:
        return []


# =========================================================
# SESSION STATE
# =========================================================

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []


if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()


if "chat_thread" not in st.session_state:
    st.session_state["chat_thread"] = []


add_thread(st.session_state["thread_id"])


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📄 Nitin Made CHATBOT")


# ---------------------------------------------------------
# NEW CHAT
# ---------------------------------------------------------

if st.sidebar.button("➕ New Chat"):

    reset_chat()

    st.rerun()


# ---------------------------------------------------------
# PDF UPLOAD
# ---------------------------------------------------------

st.sidebar.header("📚 Upload PDF")

uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)


# ---------------------------------------------------------
# PROCESS PDF
# ---------------------------------------------------------

if uploaded_file is not None:

    if st.sidebar.button("📥 Process PDF"):

        with st.spinner("Processing PDF..."):

            try:

                file_bytes = uploaded_file.read()

                result = ingest_pdf(
                    file_bytes=file_bytes,
                    thread_id=st.session_state["thread_id"],
                    filename=uploaded_file.name
                )

                st.session_state["pdf_info"] = result

                st.sidebar.success(
                    f"PDF processed successfully!"
                )

                st.sidebar.write(
                    f"**File:** {result['filename']}"
                )

                st.sidebar.write(
                    f"**Pages:** {result['documents']}"
                )

                st.sidebar.write(
                    f"**Chunks:** {result['chunks']}"
                )

            except Exception as e:

                st.sidebar.error(
                    f"Error processing PDF: {e}"
                )


# ---------------------------------------------------------
# SHOW CURRENT PDF
# ---------------------------------------------------------

if "pdf_info" in st.session_state:

    st.sidebar.divider()

    st.sidebar.subheader("📄 Current PDF")

    st.sidebar.write(
        st.session_state["pdf_info"]["filename"]
    )


# ---------------------------------------------------------
# CONVERSATION LIST
# ---------------------------------------------------------

st.sidebar.divider()

st.sidebar.header("💬 My Conversations")


for thread_id in reversed(
    st.session_state["chat_thread"]
):

    if st.sidebar.button(
        thread_id,
        key=f"thread_{thread_id}"
    ):

        st.session_state["thread_id"] = thread_id

        messages = load_conversation(thread_id)

        temp_messages = []

        for message in messages:

            if isinstance(message, HumanMessage):

                role = "user"

            elif isinstance(message, AIMessage):

                role = "assistant"

            else:

                continue

            temp_messages.append(
                {
                    "role": role,
                    "content": message.content
                }
            )

        st.session_state["message_history"] = temp_messages

        st.rerun()


# =========================================================
# MAIN UI
# =========================================================

st.title("💬 Nitin Made CHATBOT")

st.caption(
    "Upload a PDF from the sidebar and ask questions about it."
)


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state["message_history"]:

    with st.chat_message(message["role"]):

        st.write(message["content"])


# =========================================================
# CHAT INPUT
# =========================================================

user_input = st.chat_input(
    "Ask something..."
)


# =========================================================
# PROCESS USER INPUT
# =========================================================

if user_input:

    # -----------------------------------------------------
    # Display user message
    # -----------------------------------------------------

    st.session_state["message_history"].append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):

        st.write(user_input)


    # -----------------------------------------------------
    # LangGraph config
    # -----------------------------------------------------

    config = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        }
    }


    # -----------------------------------------------------
    # Generate assistant response
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        response = st.write_stream(

            message_chunk.content

            for message_chunk, metadata

            in chatbot.stream(

                {
                    "messages": [
                        HumanMessage(
                            content=user_input
                        )
                    ]
                },

                config=config,

                stream_mode="messages"
            )

            if hasattr(message_chunk, "content")
            and message_chunk.content
        )


    # -----------------------------------------------------
    # Save response
    # -----------------------------------------------------

    st.session_state["message_history"].append(
        {
            "role": "assistant",
            "content": response
        }
