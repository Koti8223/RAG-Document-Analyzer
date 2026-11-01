import os
import io
import time  # For rate limit sleep
import zipfile
import tempfile
import fitz  # PyMuPDF for image extraction
import pdfplumber  # For table extraction
from dotenv import load_dotenv

# Import LangChain's text splitter and document wrapper
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangChain Chroma and Google Embeddings libraries
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# LCEL Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Import Google GenAI to describe images
import google.generativeai as genai

# Load our API key from the .env file
load_dotenv()

class LocalMiniLMEmbeddings(Embeddings):
    """
    Zero-key, 100% local embedding model powered by Chroma's built-in ONNX all-MiniLM-L6-v2.
    Requires no API keys, runs on CPU, and has zero rate limits.
    """
    def __init__(self):
        from chromadb.utils import embedding_functions
        self.ef = embedding_functions.DefaultEmbeddingFunction()

    def embed_documents(self, texts):
        return self.ef(texts)

    def embed_query(self, text):
        return self.ef([text])[0]

def get_embedding_model(api_key: str = None):
    """
    Returns GoogleGenerativeAIEmbeddings if a Gemini key is explicitly provided,
    otherwise returns free local embeddings with zero API key requirement.
    """
    if api_key and api_key.strip():
        return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2", google_api_key=api_key.strip())
    return LocalMiniLMEmbeddings()


def convert_table_to_markdown(table):
    if not table or not table[0]:
        return ""
    cleaned_table = []
    for row in table:
        cleaned_row = [str(cell).strip().replace("\n", " ") if cell is not None else "" for cell in row]
        cleaned_table.append(cleaned_row)
    headers = cleaned_table[0]
    markdown = "\n| " + " | ".join(headers) + " |\n"
    markdown += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for row in cleaned_table[1:]:
        markdown += "| " + " | ".join(row) + " |\n"
    return markdown

def describe_image_with_gemini(image_bytes: bytes, api_key: str) -> str:
    """
    Skipped by default to preserve API quota.
    """
    return "[Image description skipped to preserve daily API quota]"

def extract_advanced_pdf(file_path: str, api_key: str = None, display_source_name: str = None):
    """
    Reads a PDF, extracts text, tables, and describes images.
    """
    documents = []
    doc_fitz = fitz.open(file_path)
    source_label = display_source_name if display_source_name else os.path.basename(file_path)
    
    with pdfplumber.open(file_path) as pdf:
        total_pages = len(pdf.pages)
        for page_idx, page in enumerate(pdf.pages):
            print(f"Processing Page {page_idx + 1}/{total_pages}...")
            
            try:
                # 1. Text Extraction
                page_text = page.extract_text() or ""
                
                # 2. Table Extraction
                tables = page.extract_tables()
                table_texts = []
                for table in tables:
                    markdown_table = convert_table_to_markdown(table)
                    if markdown_table:
                        table_texts.append(markdown_table)
                
                if table_texts:
                    page_text += "\n\n### Extracted Tables:\n" + "\n".join(table_texts)
                
                # 3. Image Extraction with PyMuPDF
                fitz_page = doc_fitz[page_idx]
                image_list = fitz_page.get_images(full=True)
                
                image_descriptions = []
                # Limit to first 3 images per page
                for img_idx, img in enumerate(image_list[:3]):
                    print(f" -> Found image {img_idx + 1} on page {page_idx + 1}. Skipping API call to save quota.")
                    try:
                        xref = img[0]
                        base_image = doc_fitz.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        description = describe_image_with_gemini(image_bytes, api_key)
                        image_descriptions.append(description)
                    except Exception as img_err:
                        print(f"Warning: Failed to extract image data on page {page_idx + 1}: {img_err}")
                    
                if image_descriptions:
                    page_text += "\n\n### Extracted Diagrams/Images:\n" + "\n".join(image_descriptions)
                    
                # Store document page
                documents.append(
                    Document(
                        page_content=page_text,
                        metadata={"source": source_label, "page": page_idx + 1}
                    )
                )
            except Exception as page_err:
                print(f"Error parsing Page {page_idx + 1}: {page_err}. Skipping page content.")
                
    doc_fitz.close()
    return documents

def extract_text_file(file_path: str, display_name: str):
    """
    Reads plain text, markdown, code, and structured data files.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []
        
    if not text.strip():
        return []
        
    return [
        Document(
            page_content=text,
            metadata={"source": display_name, "page": 1}
        )
    ]

def extract_zip_archive(zip_path: str, api_key: str = None, original_zip_name: str = "archive.zip"):
    """
    Unpacks a ZIP archive and extracts text from all enclosed PDFs, Markdown,
    text, CSV, JSON, and code files.
    """
    documents = []
    supported_text_exts = {
        ".txt", ".md", ".markdown", ".csv", ".json", ".py", ".html", 
        ".xml", ".rst", ".yaml", ".yml", ".log", ".sql"
    }
    
    with tempfile.TemporaryDirectory() as extract_dir:
        print(f"Unpacking ZIP: {original_zip_name}...")
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
        except Exception as e:
            raise ValueError(f"Failed to extract ZIP archive: {e}")
            
        for root, dirs, files in os.walk(extract_dir):
            # Ignore OS metadata and hidden directories
            dirs[:] = [d for d in dirs if not d.startswith("__") and not d.startswith(".")]
            
            for file_name in files:
                if file_name.startswith(".") or file_name.startswith("__"):
                    continue
                    
                full_path = os.path.join(root, file_name)
                rel_path = os.path.relpath(full_path, extract_dir).replace("\\", "/")
                display_name = f"{original_zip_name} / {rel_path}"
                _, ext = os.path.splitext(file_name.lower())
                
                if ext == ".pdf":
                    print(f" -> Found PDF in ZIP: {rel_path}")
                    pdf_docs = extract_advanced_pdf(full_path, api_key=api_key, display_source_name=display_name)
                    documents.extend(pdf_docs)
                elif ext in supported_text_exts:
                    print(f" -> Found text/code file in ZIP: {rel_path}")
                    txt_docs = extract_text_file(full_path, display_name)
                    documents.extend(txt_docs)
                else:
                    print(f" -> Skipping unsupported file format: {rel_path}")
                    
    return documents

def get_document_chunks(file_path: str, api_key: str = None, original_file_name: str = None):
    """
    Extracts text chunks from either a standalone PDF or a ZIP archive containing multiple documents.
    """
    name_to_check = original_file_name or file_path
    if name_to_check.lower().endswith(".zip"):
        print(f"\nProcessing ZIP archive: {name_to_check}...")
        documents = extract_zip_archive(
            file_path,
            api_key=api_key,
            original_zip_name=os.path.basename(name_to_check)
        )
    else:
        display_name = os.path.basename(name_to_check)
        documents = extract_advanced_pdf(file_path, api_key=api_key, display_source_name=display_name)
        
    if not documents:
        raise ValueError("No readable text or supported documents found in the uploaded file/archive!")
        
    print(f"\nSplitting {len(documents)} document pages/files into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks total.")
    return chunks

def initialize_vector_db(chunks, api_key: str = None, persist_directory="./chroma_db"):
    """
    Initializes Chroma vector DB using either Gemini Embeddings (if key provided)
    or fast, free LocalMiniLMEmbeddings (default, zero keys required).
    """
    print("\n--- Step 3: Initializing Embeddings & Vector DB ---")
    embeddings = get_embedding_model(api_key)
    is_local = isinstance(embeddings, LocalMiniLMEmbeddings)
    
    if is_local:
        print(f"Using 100% local MiniLM embeddings (Zero API keys required). Saving to {persist_directory}...")
    else:
        print(f"Generating Gemini embeddings and saving to {persist_directory}...")
    
    # Index in batches
    batch_size = 50
    vector_db = None
    
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        print(f" -> Indexing batch {i//batch_size + 1}...")
        
        if not is_local and i > 0:
            time.sleep(3.5)
            
        if vector_db is None:
            vector_db = Chroma.from_documents(
                documents=batch_chunks,
                embedding=embeddings,
                persist_directory=persist_directory
            )
        else:
            vector_db.add_documents(batch_chunks)
            
    print("Vector database successfully created and saved!")
    return vector_db

def query_rag_system(query: str, vector_db, api_key: str = None, groq_api_key: str = None, model_name: str = "llama-3.3-70b-versatile"):
    """
    Uses the active API key (Groq or Gemini) to run the LCEL retrieval chain.
    """
    print("\n--- Step 4: Stitching LCEL retrieval chain & generating answer ---")
    
    # Select LLM: Prefer Groq if key is provided, fallback to Gemini
    if groq_api_key and groq_api_key.strip():
        print(f"Generating answer with Groq ({model_name})...")
        llm = ChatGroq(
            model=model_name,
            groq_api_key=groq_api_key.strip(),
            temperature=0.0
        )
    elif api_key and api_key.strip():
        print("Generating answer with Gemini (gemini-2.5-flash)...")
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key.strip(),
            temperature=0.0
        )
    else:
        raise ValueError("Neither Groq API key nor Google API key was provided for text generation!")

    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    system_prompt = (
        "You are an expert research analyst. Answer the user's question using ONLY "
        "the provided context. If you don't know the answer or if it's not present in "
        "the context, say 'I don't have this information in the document.'\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])
    
    print("Searching vector database for matching chunks...")
    retrieved_docs = retriever.invoke(query)
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    generation_chain = prompt | llm | StrOutputParser()
    
    print("Generating answer...")
    answer = generation_chain.invoke({
        "context": context_text,
        "question": query
    })
    
    return {
        "answer": answer,
        "context": retrieved_docs
    }