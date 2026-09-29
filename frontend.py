import streamlit as st
from backend import chatbot
from langchain_core.messages import HumanMessage
import uuid


def generate_thread_id():
    return str(uuid.uuid4())


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_thread"]:
        st.session_state["chat_thread"].append(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )
    return state.values.get("messages", [])


if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_thread" not in st.session_state:
    st.session_state["chat_thread"] = []

add_thread(st.session_state["thread_id"])

st.title("LangGraph Chatbot")

st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()
    st.rerun()

st.sidebar.header("My Conversations")

for thread_id in st.session_state["chat_thread"][::-1]:
    if st.sidebar.button(str(thread_id), key=f"thread_{thread_id}"):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for message in messages:
            role = "user" if isinstance(message, HumanMessage) else "assistant"
            temp_messages.append(
                {"role": role, "content": message.content}
            )

        st.session_state["message_history"] = temp_messages
        st.rerun()


for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])


user_input = st.chat_input("Type here")

if user_input:
    st.session_state["message_history"].append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.write(user_input)

    config = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        }
    }

    with st.chat_message("assistant"):
        response = chatbot.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )
        answer = response["messages"][-1].content
        st.write(answer)

    st.session_state["message_history"].append(
        {"role": "assistant", "content": answer}
    )
