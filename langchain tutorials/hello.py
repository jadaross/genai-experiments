from langchain_ollama import OllamaLLM, ChatOllama
from langchain_community import OllamaEmbeddings

llm = OllamaLLM(model="llama3.2")
embeddings= OllamaEmbeddings(model="llama3.2")


#print(llm.invoke("The first man on the moon was ..."))
#chat_model = ChatOllama(model="llama3.2")
#print(chat_model.invoke("Who was the first man on the moon?"))

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Only keep post title, headers, and content from the full HTML.
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

assert len(docs) == 1
print(f"Total characters: {len(docs[0].page_content)}")
print(docs[0].page_content[:500])

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)

print(f"Split blog post into {len(all_splits)} sub-documents.")

from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)

document_ids = vector_store.add_documents(documents=all_splits)

print(document_ids[:3])