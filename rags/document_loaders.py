from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin, urlparse
from typing import Set, List
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.storage import InMemoryByteStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from datetime import datetime
from uuid import uuid4
import uuid

def filter_executive_orders(urls: Set[str]) -> List[Document]:
    """
    Convert filtered URLs into RAG-suitable document objects.
    This function processes a set of URLs, loads the content from each URL, and filters out documents
    that do not mention "executive order". It extracts the title and content of each document, checks
    for mentions of "executive order" in both the metadata and content, and creates a list of document
    objects with enhanced metadata for those that do.
    Args:
        urls (Set[str]): A set of URLs to be processed.
    Returns:
        List[Document]: A list of Document objects that mention "executive order".
    """
    rag_documents = []
    
    for url in urls:
        try:
            loader = RecursiveUrlLoader(
                url=url,
                max_depth=1,
                extractor=extract_title_and_content,
                prevent_outside=True,
                timeout=10
            )
            
            docs = loader.load()
            if not docs:
                continue
                
            doc = docs[0]
            
            # Extract title and content
            parts = doc.page_content.split("\n\n")
            title = parts[0].replace("Title: ", "")
            
            # Skip fact sheets
            if "fact sheet" in title.lower():
                continue
                
            # Check for executive order mentions
            has_executive_order = False
            locations = []
            
            # Check metadata
            for key, value in doc.metadata.items():
                if isinstance(value, str) and "executive order" in value.lower():
                    has_executive_order = True
                    locations.append(f"metadata.{key}")
            
            # Check content
            content = "\n\n".join(parts[1:])
            if "executive order" in content.lower():
                has_executive_order = True
                locations.append("content")
            
            if has_executive_order:
                # Create enhanced metadata
                enhanced_metadata = {
                    **doc.metadata,  # Original metadata
                    "executive_order_locations": locations,
                    "source_type": "executive_order",
                    "extraction_date": datetime.now().isoformat()
                }
                
                # Create document object
                rag_doc = Document(
                    page_content=content,
                    metadata={
                        "title": title,
                        "url": url,
                        "date": None,  # Will be set in post_init
                        **enhanced_metadata
                    }
                )
                
                rag_documents.append(rag_doc)
                
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
    
    return rag_documents

def get_all_presidential_actions(base_url: str, max_depth: int = 3) -> Set[str]:
    """
    Get all matching presidential action URLs from the given base URL up to a specified depth.
    This function crawls the given base URL and extracts all URLs that match the target pattern
    for presidential actions. It performs a recursive search up to the specified maximum depth.
    Args:
        base_url (str): The base URL to start crawling from.
        max_depth (int, optional): The maximum depth to crawl. Defaults to 3.
    Returns:
        Set[str]: A set of URLs that match the target pattern for presidential actions.
    Raises:
        Exception: If there is an error processing a URL, it will be printed to the console.
    """
    visited = set()
    all_matched_urls = set()
    base_domain = urlparse(base_url).netloc
    target_pattern = "https://www.whitehouse.gov/presidential-actions/2025/01/"
    
    def recursive_extract(url: str, current_depth: int = 0):
        if current_depth > max_depth:
            return
            
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                absolute_url = urljoin(url, href)
                
                if urlparse(absolute_url).netloc != base_domain:
                    continue
                    
                if (absolute_url.startswith(target_pattern) and 
                    not absolute_url.endswith('#top') and
                    absolute_url not in all_matched_urls):
                    all_matched_urls.add(absolute_url)
                
                if absolute_url not in visited:
                    visited.add(absolute_url)
                    recursive_extract(absolute_url, current_depth + 1)
                    
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
    
    recursive_extract(base_url)
    return all_matched_urls

def extract_title_and_content(html):
    """
    Extracts the title and main content from an HTML document.
    This function parses the provided HTML string to extract the text content
    of the first <h1> element as the title and the text content of the main 
    content area (identified by <main>, <article>, or a <div> with class 'content') 
    as the content. Unwanted elements such as <nav>, <footer>, <script>, and <style> 
    within the main content area are removed.
    Args:
        html (str): A string containing the HTML content to be parsed.
    Returns:
        str: A formatted string containing the extracted title and content.
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    title = ""
    title_elem = soup.find('h1')
    if title_elem:
        title = title_elem.get_text(strip=True)
    
    content = ""
    main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
    if main_content:
        for unwanted in main_content.find_all(['nav', 'footer', 'script', 'style']):
            unwanted.decompose()
        content = main_content.get_text(strip=True, separator=' ')
    
    return f"Title: {title}\n\nContent: {content}"

def load_documents(start_url):
    """
    Load and process documents starting from the given URL.

    This function retrieves all URLs related to presidential actions starting from the provided URL,
    filters them to include only executive orders, and returns the filtered documents.

    Args:
        start_url (str): The starting URL to begin retrieving presidential actions.

    Returns:
        list: A list of filtered documents containing only executive orders.
    """

    all_urls = get_all_presidential_actions(start_url)
    docs = filter_executive_orders(all_urls)

    print(f"\nProcessed {len(docs)} documents for RAG system:")
    return docs

def vectorise_documents(docs: List[Document], vector_store):
    """
    Vectorizes a list of documents and stores them in a vector store for retrieval.

    This function takes a list of documents, summarizes them using a language model,
    and stores both the original documents and their summaries in a vector store.
    The summaries are generated using a chain of operations that includes a prompt
    template and a language model.

    Args:
        docs (List[Document]): A list of Document objects to be vectorized and summarized.
        vector_store: The vector store where the summarized documents will be stored.

    Returns:
        MultiVectorRetriever: A retriever object that can be used to retrieve documents
        from the vector store based on their summaries.
    """

    chain = (
    {"doc": lambda x: x.page_content}
    | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
    | ChatOpenAI(model="gpt-3.5-turbo",max_retries=0)
    | StrOutputParser()
    )   

    summaries = chain.batch(docs, {"max_concurrency": 5})
    store = InMemoryByteStore()
    id_key = "doc_id"

    # The retriever
    retriever = MultiVectorRetriever(
        vectorstore=vector_store,
        byte_store=store,
        id_key=id_key,
    )
    doc_ids = [str(uuid.uuid4()) for _ in docs]

    # Docs linked to summaries
    summary_docs = [
        Document(page_content=s, metadata={id_key: doc_ids[i]})
        for i, s in enumerate(summaries)
    ]

    # Add
    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, docs)))

    return retriever

def vectorise_documents_basic(docs: List[Document], vector_store):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )

    # Process documents and create chunks with metadata
    for doc in docs:
        doc_id = str(uuid4())  # Generate unique ID for each document
        chunks = text_splitter.create_documents(
            texts=[doc.page_content],
            metadatas=[{
                "title": doc.metadata["title"],
                "date": str(doc.metadata["date"]),  # Convert datetime to string
                "url": doc.metadata["url"],
                "document_id": doc_id,
                "chunk_number": i  # Add chunk number for ordering
            } for i in range(len(text_splitter.split_text(doc.page_content)))]
        )
        # Add chunks to vector store
        vector_store.add_documents(chunks)
    
    return vector_store