from langchain_ollama import OllamaLLM, ChatOllama
import getpass
import os

# LangGraph
#os.environ["LANGCHAIN_TRACING_V2"] = "true"
#os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()

llm = OllamaLLM(model="llama3.2")
print(llm.invoke("Can you write a tutorial for me on how to use langsmith?"))
