# Basic Imports
from dotenv import load_dotenv
from typing import List
import sys
import os

# Langchain Community Imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain import hub

# Set the correct directory for the project
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(parent_directory)
sys.path.append(parent_directory)

# Local function imports
from rags.document_loaders import load_documents, vectorise_documents

# Load in environment variable and set a project name for langsmith tracing
load_dotenv("env.yaml")
os.environ["LANGSMITH_PROJECT"] = "executive-order-rag-testing"

# Load in LLM, Embeddings, and Vector Store
llm = ChatOpenAI(model="gpt-4o")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)
start_url = "https://www.whitehouse.gov/presidential-actions/"

# Reading in all presidential actions from the White House Gov website, filtering to executive orders and processing into document objects
docs = load_documents(start_url=start_url)
retriever = vectorise_documents(docs=docs, vector_store=vector_store)
prompt = hub.pull("rlm/rag-prompt")

# Create LangGraph
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):

    template = """You are an AI language model assistant. Your task is to generate three 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines. Original question: {question}"""

    prompt_perspectives = ChatPromptTemplate.from_template(template)

    generate_queries = (
        prompt_perspectives 
        | ChatOpenAI(temperature=0) 
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
    )

    def get_unique_union(documents: list[list]):
        """ Unique union of retrieved docs """
        # Flatten list of lists, and convert each Document to string
        flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
        # Get unique documents
        unique_docs = list(set(flattened_docs))
        # Return
        return [loads(doc) for doc in unique_docs]

    # Retrieve
    retrieval_chain = generate_queries | retriever.map() | get_unique_union
    retrieved_docs = retrieval_chain.invoke({"question":state["question"]})

    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# Build the graph and compile with memory
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# Invoke
result = graph.invoke({"question": "What executive orders are about women?"})
print(f'Answer: {result["answer"]}\n\n')


