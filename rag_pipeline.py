import os
import io
import time  # For rate limit sleep
import fitz  # PyMuPDF for image extraction
import pdfplumber  # For table extraction
from dotenv import load_dotenv

# Import LangChain's text splitter and document wrapper
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangChain Chroma and Google Embeddings libraries
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# LCEL Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Import Google GenAI to describe images
import google.generativeai as genai

# Load our API key from the .env file
load_dotenv()

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

def extract_advanced_pdf(file_path: str, api_key: str):
    """
    Reads a PDF, extracts text, tables, and describes images.
    """
    documents = []
    doc_fitz = fitz.open(file_path)
    
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
                        metadata={"source": file_path, "page": page_idx + 1}
                    )
                )
            except Exception as page_err:
                print(f"Error parsing Page {page_idx + 1}: {page_err}. Skipping page content.")
                
    doc_fitz.close()
    return documents

def get_document_chunks(file_path: str, api_key: str):
    documents = extract_advanced_pdf(file_path, api_key)
    print("\nSplitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks total.")
    return chunks

def initialize_vector_db(chunks, api_key: str, persist_directory="./chroma_db"):
    """
    Uses the active API key to generate embeddings and initialize Chroma.
    """
    print("\n--- Step 3: Initializing Embeddings & Vector DB ---")
    if not api_key:
        raise ValueError("Google API key is missing! Please provide a key in the sidebar.")
    
    # Use the active api_key dynamically
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2", google_api_key=api_key)
    print(f"Generating embeddings and saving to {persist_directory}...")
    
    # Index in batches
    batch_size = 50
    vector_db = None
    
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        print(f" -> Indexing batch {i//batch_size + 1}...")
        
        if i > 0:
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

def query_rag_system(query: str, vector_db, api_key: str):
    """
    Uses the active API key to run the Gemini LCEL retrieval chain.
    """
    print("\n--- Step 4: Stitching LCEL retrieval chain & generating answer ---")
    if not api_key:
        raise ValueError("Google API key is missing! Please provide a key in the sidebar.")

    # Initialize Gemini with the active API key
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0.0)
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
    
    print("Generating answer with Gemini...")
    answer = generation_chain.invoke({
        "context": context_text,
        "question": query
    })
    
    return {
        "answer": answer,
        "context": retrieved_docs
    }