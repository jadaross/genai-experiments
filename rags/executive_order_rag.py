# Basic Imports
from dotenv import load_dotenv
from typing import List
from uuid import uuid4
import sys
import os

# Langchain Community Imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langchain import hub

# Set the correct directory for the project
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(parent_directory)
sys.path.append(parent_directory)

# Local function imports
from rags.document_loaders import get_all_presidential_actions, filter_executive_orders, load_documents

# Load in environment variable and set a project name for langsmith tracing
load_dotenv("env.yaml")
os.environ["LANGSMITH_PROJECT"] = "test-rag-using-lang-smith-jada-ross"

# Load in LLM, Embeddings, and Vector Store
llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)
start_url = "https://www.whitehouse.gov/presidential-actions/"

# Reading in all presidential actions from the White House Gov website, filtering to executive orders and processing into document objects
vector_store = load_documents(embeddings = embeddings, vector_store = vector_store, start_url = start_url)

prompt = hub.pull("rlm/rag-prompt")

example_messages = prompt.invoke(
    {"context": "(context goes here)", "question": "(question goes here)"}
).to_messages()

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# Specify an ID for the thread so the RAG system can keep track of the conversation and initiate a Memory object
config = {"configurable": {"thread_id": "test-women"}}
memory = MemorySaver()

# Build the graph and compile with memory
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile(checkpointer=memory)

# First question
result = graph.invoke({"question": "What executive orders are about women?"}, config=config)
print(f'Context: {result["context"]}\n\n')
print(f'Answer: {result["answer"]}\n\n')


