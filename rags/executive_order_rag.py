# Basic Imports
from dotenv import load_dotenv
from typing import List
from uuid import uuid4
import sys
import os

# Langchain Community Imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain import hub

# Local function imports
from rags.document_loaders import get_all_presidential_actions, filter_executive_orders

# Set the correct directory for the project
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(parent_directory)
sys.path.append(parent_directory)

# Load in environment variable and set a project name for langsmith tracing
load_dotenv("env.yaml")
os.environ["LANGSMITH_PROJECT"] = "test-rag-using-lang-smith-jada-ross"

# Load in LLM, Embeddings, and Vector Store
llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)

# Reading in all presidential actions from the White House Gov website, filtering to executive orders and processing into document objects
start_url = "https://www.whitehouse.gov/presidential-actions/"
all_urls = get_all_presidential_actions(start_url)
rag_docs = filter_executive_orders(all_urls)

print(f"\nProcessed {len(rag_docs)} documents for RAG system:")

##### INDEXING 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

# Process documents and create chunks with metadata
all_chunks = []
for doc in rag_docs:
    doc_id = str(uuid4())  # Generate unique ID for each document
    chunks = text_splitter.create_documents(
        texts=[doc.content],
        metadatas=[{
            "title": doc.title,
            "date": str(doc.date),  # Convert datetime to string
            "url": doc.url,
            "document_id": doc_id,
            "chunk_number": i  # Add chunk number for ordering
        } for i in range(len(text_splitter.split_text(doc.content)))]
    )
    # Add chunks to vector store
    vector_store.add_documents(chunks)

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

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

result = graph.invoke({"question": "What is the latest executive order created by Trump?"})

print(f'Context: {result["context"]}\n\n')
print(f'Answer: {result["answer"]}')