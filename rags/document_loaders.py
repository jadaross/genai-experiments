from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin, urlparse
from typing import Set, Tuple, List
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
import re
from uuid import uuid4

@dataclass
class ExecutiveOrderDocument:
    """Class to store executive order documents in a format suitable for RAG"""
    title: str
    url: str
    content: str
    date: Optional[datetime]
    metadata: Dict
    
    def __post_init__(self):
        """Extract and parse date from URL after initialization"""
        date_match = re.search(r'/(\d{4})/(\d{2})/', self.url)
        if date_match:
            year, month = date_match.groups()
            self.date = datetime(int(year), int(month), 1)  # Using 1st of month as default day
            
    def to_dict(self) -> Dict:
        """Convert document to dictionary format"""
        return {
            "title": self.title,
            "url": self.url,
            "content": self.content,
            "date": self.date.isoformat() if self.date else None,
            "metadata": self.metadata
        }

def filter_executive_orders(urls: Set[str]) -> List[ExecutiveOrderDocument]:
    """Convert filtered URLs into RAG-suitable document objects"""
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
                rag_doc = ExecutiveOrderDocument(
                    title=title,
                    url=url,
                    content=content,
                    date=None,  # Will be set in post_init
                    metadata=enhanced_metadata
                )
                
                rag_documents.append(rag_doc)
                
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
    
    return rag_documents

def get_all_presidential_actions(base_url: str, max_depth: int = 3) -> Set[str]:
    """Get all matching presidential action URLs"""
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

    all_urls = get_all_presidential_actions(start_url)
    docs = filter_executive_orders(all_urls)

    print(f"\nProcessed {len(docs)} documents for RAG system:")
    return docs

def vectorise_documents(docs: List[ExecutiveOrderDocument], vector_store):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )

    # Process documents and create chunks with metadata
    for doc in docs:
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
    
    return vector_store