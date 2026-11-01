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
    Reads text, markdown, code, and structured CSV/TSV/JSON files safely without blowing memory.
    """
    _, ext = os.path.splitext(file_path.lower())
    file_size = os.path.getsize(file_path)
    
    # Special handling for CSV / TSV tabular datasets
    if ext in [".csv", ".tsv"]:
        try:
            import csv
            delimiter = "\t" if ext == ".tsv" else ","
            rows = []
            total_rows = 0
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f, delimiter=delimiter)
                for i, row in enumerate(reader):
                    total_rows += 1
                    if i < 25:
                        rows.append(row)
                # Count the rest if file is large
                for _ in f:
                    total_rows += 1
            
            if rows:
                headers = rows[0]
                md = f"### Dataset Profile: `{display_name}`\n"
                md += f"- **Format:** {ext.upper()[1:]} Tabular Data\n"
                md += f"- **Estimated Rows:** ~{total_rows:,}\n"
                md += f"- **Total Columns:** {len(headers)}\n"
                md += f"- **Column Names:** `" + "`, `".join(str(h) for h in headers[:40]) + "`\n\n"
                md += "#### Sample Data Preview (First Rows):\n"
                md += convert_table_to_markdown(rows[:15])
                return [
                    Document(
                        page_content=md,
                        metadata={"source": display_name, "page": 1}
                    )
                ]
        except Exception as e:
            print(f"Error reading tabular file {file_path}: {e}")

    # For text, markdown, json, code files
    try:
        # Read up to 120 KB to prevent memory exhaustion
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read(120_000)
            if file_size > 120_000:
                text += f"\n\n[... File truncated: showing first 120KB of {file_size / 1024:.1f}KB ...]"
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
    Unpacks a ZIP archive, inspects all files (including datasets like .h5, .npz, .pkl, .csv),
    extracts readable content from PDFs/text/code, and generates a structured
    Manifest Document so no ZIP archive ever returns empty.
    """
    documents = []
    supported_text_exts = {
        ".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".py", ".html", 
        ".xml", ".rst", ".yaml", ".yml", ".log", ".sql", ".ini", ".toml", ".cfg",
        ".sh", ".bat", ".ipynb"
    }
    dataset_binary_exts = {
        ".h5", ".hdf5", ".npz", ".npy", ".pkl", ".pickle", ".parquet", ".feather",
        ".mat", ".pt", ".pth", ".bin", ".dat"
    }
    
    manifest_entries = []
    total_uncompressed_bytes = 0
    
    with tempfile.TemporaryDirectory() as extract_dir:
        print(f"Unpacking ZIP: {original_zip_name}...")
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
        except Exception as e:
            raise ValueError(f"Failed to extract ZIP archive: {e}")
            
        for root, dirs, files in os.walk(extract_dir):
            # Ignore hidden directories & cache
            dirs[:] = [d for d in dirs if not d.startswith("__") and not d.startswith(".")]
            
            for file_name in files:
                if file_name.startswith(".") or file_name.startswith("__"):
                    continue
                    
                full_path = os.path.join(root, file_name)
                rel_path = os.path.relpath(full_path, extract_dir).replace("\\", "/")
                display_name = f"{original_zip_name} / {rel_path}"
                file_size = os.path.getsize(full_path)
                total_uncompressed_bytes += file_size
                _, ext = os.path.splitext(file_name.lower())
                
                size_str = f"{file_size / (1024*1024):.2f} MB" if file_size >= 1024*1024 else f"{file_size / 1024:.1f} KB"
                manifest_entries.append({"path": rel_path, "size": size_str, "ext": ext or "unknown"})
                
                if ext == ".pdf":
                    print(f" -> Found PDF in ZIP: {rel_path}")
                    pdf_docs = extract_advanced_pdf(full_path, api_key=api_key, display_source_name=display_name)
                    documents.extend(pdf_docs)
                elif ext in supported_text_exts:
                    print(f" -> Found text/data file in ZIP: {rel_path}")
                    txt_docs = extract_text_file(full_path, display_name)
                    documents.extend(txt_docs)
                elif ext in dataset_binary_exts:
                    print(f" -> Found dataset file in ZIP: {rel_path} ({size_str})")
                    doc_content = (
                        f"### Dataset File: `{rel_path}`\n"
                        f"- **Archive:** `{original_zip_name}`\n"
                        f"- **Format:** `{ext}` binary scientific / ML dataset\n"
                        f"- **Size:** {size_str}\n"
                        f"- **Description:** Enclosed data file in the `{original_zip_name}` package."
                    )
                    documents.append(
                        Document(
                            page_content=doc_content,
                            metadata={"source": display_name, "page": 1}
                        )
                    )
                else:
                    print(f" -> Enclosed file in ZIP: {rel_path} ({size_str})")
                    
    # Always build the comprehensive archive manifest document
    total_mb = total_uncompressed_bytes / (1024 * 1024)
    manifest_md = f"# Archive Overview & File Manifest: `{original_zip_name}`\n"
    manifest_md += f"- **Archive Name:** `{original_zip_name}`\n"
    manifest_md += f"- **Total Enclosed Files:** {len(manifest_entries)}\n"
    manifest_md += f"- **Total Uncompressed Size:** {total_mb:.2f} MB\n\n"
    manifest_md += "### Enclosed Files & Directory Structure:\n"
    for entry in manifest_entries[:100]:
        manifest_md += f"- `{entry['path']}` (Size: {entry['size']}, Type: {entry['ext']})\n"
    if len(manifest_entries) > 100:
        manifest_md += f"\n*...and {len(manifest_entries) - 100} additional files.*"
        
    manifest_doc = Document(
        page_content=manifest_md,
        metadata={"source": f"{original_zip_name} / [Archive Manifest]", "page": 1}
    )
    documents.insert(0, manifest_doc)
                    
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
        # Fallback safeguard: Never allow empty document set
        documents = [
            Document(
                page_content=f"Uploaded file: {os.path.basename(name_to_check)} (No parseable text detected).",
                metadata={"source": os.path.basename(name_to_check), "page": 1}
            )
        ]
        
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

def query_rag_system(query: str, vector_db, api_key: str = None, groq_api_key: str = None, model_name: str = "llama-3.1-8b-instant"):
    """
    Uses the active API key (Groq or Gemini) to run the LCEL retrieval chain.
    Automatically handles model fallbacks if a specific model ID is unavailable (e.g. 404).
    """
    print("\n--- Step 4: Stitching LCEL retrieval chain & generating answer ---")
    
    clean_groq_key = groq_api_key.strip().strip("'").strip('"') if groq_api_key else ""
    clean_gemini_key = api_key.strip().strip("'").strip('"') if api_key else ""

    if not clean_groq_key and not clean_gemini_key:
        raise ValueError("No API key provided. Please enter your Groq API key in the sidebar.")

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
    
    answer = None
    last_error = None
    
    # 1. Try Groq with auto-fallback across production models
    if clean_groq_key:
        candidate_models = [model_name, "llama-3.1-8b-instant", "llama3-8b-8192", "mixtral-8x7b-32768"]
        # Remove duplicates while keeping order
        ordered_models = []
        for m in candidate_models:
            if m and m not in ordered_models:
                ordered_models.append(m)
                
        for target_model in ordered_models:
            try:
                print(f"Generating answer with Groq ({target_model})...")
                llm = ChatGroq(
                    model=target_model,
                    groq_api_key=clean_groq_key,
                    temperature=0.0
                )
                generation_chain = prompt | llm | StrOutputParser()
                answer = generation_chain.invoke({
                    "context": context_text,
                    "question": query
                })
                if answer:
                    break
            except Exception as e:
                err_str = str(e)
                last_error = e
                # If 404 (model not found / no access), try next candidate model
                if "404" in err_str or "model_not_found" in err_str.lower() or "does not exist" in err_str.lower():
                    print(f"Notice: Model {target_model} not accessible, attempting fallback...")
                    continue
                else:
                    # Non-404 error (e.g. 401 invalid key, rate limit)
                    raise e

    # 2. Try Gemini fallback if Groq failed or wasn't provided
    if not answer and clean_gemini_key:
        try:
            print("Generating answer with Gemini (gemini-2.5-flash)...")
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=clean_gemini_key,
                temperature=0.0
            )
            generation_chain = prompt | llm | StrOutputParser()
            answer = generation_chain.invoke({
                "context": context_text,
                "question": query
            })
        except Exception as e:
            last_error = e

    if not answer:
        if last_error:
            raise last_error
        raise ValueError("Could not generate answer. Please check your API key.")

    return {
        "answer": answer,
        "context": retrieved_docs
    }