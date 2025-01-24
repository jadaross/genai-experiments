import chainlit as cl
from dotenv import load_dotenv
import os
import sys
from typing import List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict
from langchain import hub

# Set the correct directory for the project
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(parent_directory)
sys.path.append(parent_directory)

# Local function imports
from rags.document_loaders import load_documents, vectorise_documents

# Load environment variables
load_dotenv("env.yaml")
os.environ["LANGSMITH_PROJECT"] = "executive-order-rag-streamlit"

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    sources: List[str]  # Added sources to track document metadata

@cl.on_chat_start
async def on_chat_start():
    # Initialize your RAG components
    llm = ChatOpenAI(model="gpt-4o")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = InMemoryVectorStore(embeddings)
    start_url = "https://www.whitehouse.gov/presidential-actions/"

    # Load and process documents
    docs = load_documents(start_url=start_url)
    retriever = vectorise_documents(docs=docs, vector_store=vector_store)
    prompt = hub.pull("rlm/rag-prompt")

    # Store the components in the user session
    cl.user_session.set("retriever", retriever)
    cl.user_session.set("llm", llm)
    cl.user_session.set("prompt", prompt)

    await cl.Message(content="Ready to answer questions about executive orders! How can I help?").send()

def retrieve(state: State):
    retriever = cl.user_session.get("retriever")
    
    template = """You are an AI language model assistant. Your task is to generate three 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines. Original question: {question}"""

    prompt_perspectives = ChatPromptTemplate.from_template(template)
    llm = cl.user_session.get("llm")

    generate_queries = (
        prompt_perspectives 
        | llm
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
    )

    def get_unique_union(documents: list[list]):
        flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
        unique_docs = list(set(flattened_docs))
        return [loads(doc) for doc in unique_docs]

    retrieval_chain = generate_queries | retriever.map() | get_unique_union
    retrieved_docs = retrieval_chain.invoke({"question": state["question"]})

    # Extract source information from documents
    sources = []
    for doc in retrieved_docs:
        if hasattr(doc, 'metadata') and 'source' in doc.metadata:
            sources.append(doc.metadata['source'])
        elif hasattr(doc, 'metadata') and 'url' in doc.metadata:
            sources.append(doc.metadata['url'])
        else:
            sources.append("Unknown source")

    return {"context": retrieved_docs, "sources": list(set(sources))}  # Deduplicate sources

def generate(state: State):
    llm = cl.user_session.get("llm")
    prompt = cl.user_session.get("prompt")
    
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

@cl.on_message
async def main(message: cl.Message):
    # Build the graph
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    # Show thinking message
    thinking_msg = cl.Message(content="Searching through documents...")
    await thinking_msg.send()

    # Get response from graph
    result = graph.invoke({"question": message.content})
    
    # Create the response with sources
    response_content = f"{result['answer']}\n\nSources:\n"
    for source in result.get('sources', []):
        response_content += f"- {source}\n"

    # Update thinking message with final response and sources
    thinking_msg.content = response_content
    await thinking_msg.update()

# Run the app
if __name__ == "__main__":
    cl.run()