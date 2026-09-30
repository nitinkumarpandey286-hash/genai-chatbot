
import os
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict

from dotenv import load_dotenv

# LangChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

# Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Chat model
from langchain_openai import ChatOpenAI

# LangGraph
from langchain_core.messages import BaseMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY is not set. "
        "Please add it to your .env file."
    )


# =========================================================
# CHAT MODEL
# =========================================================

model = ChatOpenAI(
    model="openrouter/free",
    temperature=0.7,
    max_tokens=2000,
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)


# =========================================================
# EMBEDDING MODEL
# =========================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =========================================================
# THREAD STORAGE
# =========================================================

THREAD_RETRIEVERS: Dict[str, Any] = {}

_THREAD_METADATA: Dict[str, dict] = {}


# =========================================================
# GET RETRIEVER
# =========================================================

def _get_retriever(thread_id: Optional[str]):
    """
    Fetch the retriever for a thread if available.
    """

    if thread_id and thread_id in THREAD_RETRIEVERS:
        return THREAD_RETRIEVERS[thread_id]

    return None


# =========================================================
# PDF INGESTION
# =========================================================

def ingest_pdf(
    file_bytes: bytes,
    thread_id: str,
    filename: Optional[str] = None
) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF
    and store it for the current thread.
    """

    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    # Create temporary PDF file
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:

        # Load PDF
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        # Split PDF into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=[
                "\n\n",
                "\n",
                " ",
                ""
            ]
        )

        chunks = splitter.split_documents(docs)

        if not chunks:
            raise ValueError("No text could be extracted from the PDF.")

        # Create FAISS vector database
        vector_store = FAISS.from_documents(
            chunks,
            embeddings
        )

        # Create retriever
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 4
            }
        )

        # Store retriever for this thread
        THREAD_RETRIEVERS[str(thread_id)] = retriever

        # Store metadata
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

    finally:

        # Delete temporary PDF
        try:
            os.remove(temp_path)

        except OSError:
            pass


# =========================================================
# LANGGRAPH STATE
# =========================================================

class ChatState(TypedDict):
    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]


# =========================================================
# CHAT NODE
# =========================================================

def chat_node(state: ChatState):

    messages = state["messages"]

    response = model.invoke(messages)

    return {
        "messages": [response]
    }


# =========================================================
# MEMORY
# =========================================================

checkpointer = MemorySaver()


# =========================================================
# CREATE GRAPH
# =========================================================

graph = StateGraph(ChatState)

graph.add_node(
    "chat_node",
    chat_node
)

graph.add_edge(
    START,
    "chat_node"
)

graph.add_edge(
    "chat_node",
    END
)


# =========================================================
# COMPILE CHATBOT
# =========================================================

chatbot = graph.compile(
    checkpointer=checkpointer
)



