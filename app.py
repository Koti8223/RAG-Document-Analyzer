try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import os
import re
import streamlit as st
import tempfile
from dotenv import load_dotenv

# Import our backend functions
from rag_pipeline import get_document_chunks, initialize_vector_db, query_rag_system, get_embedding_model
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="DeepDoc - Research Paper Analyzer",
    page_icon="🧠",
    layout="wide"
)

# Initialize Streamlit Session State (Memory)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "current_file" not in st.session_state:
    st.session_state.current_file = None

if "indexing_error" not in st.session_state:
    st.session_state.indexing_error = None

def display_source_tabs(context):
    """
    Renders source citations inside a clean, modern tabbed interface.
    """
    if not context:
        return
        
    st.markdown("### 🔍 Source Citations")
    tabs = st.tabs([f"📄 Source {i+1} ({os.path.basename(doc.metadata.get('source', 'Unknown'))})" for i, doc in enumerate(context)])
    
    for idx, tab in enumerate(tabs):
        doc = context[idx]
        with tab:
            source_name = doc.metadata.get("source", "Unknown File")
            page_num = doc.metadata.get("page", "N/A")
            st.caption(f"📂 **Source:** `{source_name}` | 📄 **Page:** `{page_num}`")
            formatted_text = "\n> ".join(doc.page_content.split("\n"))
            st.markdown(f"> {formatted_text}")

# --- SIDEBAR (For uploading and settings) ---
with st.sidebar:
    st.title("⚙️ Document Settings")
    
    # Check default system keys configured in env/secrets
    system_groq_key = os.getenv("GROQ_API_KEY")
    system_groq_key_exists = system_groq_key is not None and system_groq_key.strip() != ""
    
    # 1. Groq API Key (Primary LLM generation key)
    groq_label = "⚡ Groq API Key (Optional - In .env)" if system_groq_key_exists else "⚡ Groq API Key (Required)"
    groq_placeholder = "Uses GROQ_API_KEY from .env..." if system_groq_key_exists else "Enter Groq Key (gsk_...)"
    user_groq_key = st.text_input(
        groq_label,
        type="password",
        placeholder=groq_placeholder,
        help="Get your key from console.groq.com. Powers ultra-fast answer generation."
    )
    active_groq_key = user_groq_key.strip() if user_groq_key and user_groq_key.strip() else (system_groq_key.strip() if system_groq_key else "")
    
    # Groq Model selection
    groq_model_options = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it"
    ]
    selected_model = st.selectbox(
        "🧠 LLM Model",
        options=groq_model_options,
        index=0,
        help="Select the Groq model used to generate answers."
    )
    
    if active_groq_key:
        st.caption("⚡ Groq LPU active for high-speed generation.")
    else:
        st.caption("ℹ️ Please enter your Groq API key above.")
        
    # Optional Gemini settings for advanced users
    system_gemini_key = os.getenv("GOOGLE_API_KEY")
    with st.expander("⚙️ Optional Settings (Gemini Fallback)", expanded=False):
        user_gemini_key = st.text_input(
            "Gemini API Key (Optional)",
            type="password",
            placeholder="Uses GOOGLE_API_KEY from .env if present...",
            help="Optional. If omitted, DeepDoc automatically uses free local embeddings."
        )
    active_gemini_key = user_gemini_key.strip() if user_gemini_key and user_gemini_key.strip() else (system_gemini_key.strip() if system_gemini_key else "")
    
    st.write("---")
    
    uploaded_file = st.file_uploader(
        "Upload a Document or Archive (PDF, ZIP)", 
        type=["pdf", "zip"], 
        help="Upload any PDF or ZIP archive containing PDFs, markdown, text, or code files."
    )
    
    # CASE A: A file is actively uploaded
    if uploaded_file is not None:
        if not active_groq_key and not active_gemini_key:
            st.error("🔑 Please enter your Groq API Key in the sidebar to proceed!")
        else:
            if uploaded_file.name != st.session_state.current_file:
                st.session_state.current_file = uploaded_file.name
                st.session_state.chat_history = []  # Clear history for new file
                st.session_state.indexing_error = None
                
                safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', uploaded_file.name)
                db_dir = f"./chroma_db_{safe_name}"
                
                with st.status(f"Indexing {uploaded_file.name}...", expanded=True) as status:
                    try:
                        embeddings = get_embedding_model(active_gemini_key)
                        
                        # Performance: Check if the vector database already exists on disk (Caching)
                        if os.path.exists(db_dir):
                            status.write("⚡ Found existing database on disk. Loading instantly...")
                            st.session_state.vector_db = Chroma(
                                persist_directory=db_dir,
                                embedding_function=embeddings
                            )
                            st.session_state.indexing_error = None
                            status.update(label="Database loaded from cache!", state="complete", expanded=False)
                            st.success("Loaded from disk instantly! Ready for Q&A.")
                        else:
                            file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                tmp_file_path = tmp_file.name
                            
                            status.write(f"Extracting contents from {uploaded_file.name}...")
                            chunks = get_document_chunks(
                                tmp_file_path, 
                                api_key=active_gemini_key,
                                original_file_name=uploaded_file.name
                            )
                            
                            status.write(f"Building vector database for {len(chunks)} text chunks...")
                            st.session_state.vector_db = initialize_vector_db(
                                chunks, 
                                api_key=active_gemini_key, 
                                persist_directory=db_dir
                            )
                            st.session_state.indexing_error = None
                            
                            os.remove(tmp_file_path)
                            status.update(label="Document(s) indexed successfully!", state="complete", expanded=False)
                            st.success(f"Ready for Q&A! ({len(chunks)} chunks indexed)")
                    except Exception as e:
                        status.update(label="Indexing failed!", state="error")
                        st.error(f"Error: {e}")
                        st.session_state.vector_db = None
                        st.session_state.indexing_error = str(e)

                    
    # CASE B: No file is uploaded
    else:
        if st.session_state.current_file is not None:
            st.session_state.current_file = None
            st.session_state.vector_db = None
            st.session_state.indexing_error = None
            st.session_state.chat_history = []
            st.rerun()

    # --- SIDEBAR SETTINGS ---
    if st.session_state.vector_db:
        st.write("---")
        st.info(f"Active Document:\n**{st.session_state.current_file}**")
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# --- MAIN SCREEN ---
st.title("🧠 DeepDoc: Advanced Document Analyzer")
st.write("Ask questions and extract concepts or data from text, tables, and diagrams instantly.")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "context" in message:
            with st.expander("🔍 View Source Citations"):
                display_source_tabs(message["context"])

# Chat Input Box
if query := st.chat_input("Ask a question about the document..."):
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.chat_history.append({"role": "user", "content": query})
    
    if not st.session_state.vector_db:
        with st.chat_message("assistant"):
            if st.session_state.get("indexing_error"):
                st.error(f"⚠️ Could not index **{st.session_state.current_file}**: {st.session_state.indexing_error}")
            else:
                st.warning("Please upload a document or archive (PDF or ZIP) in the sidebar first!")
    elif not active_groq_key and not active_gemini_key:
        with st.chat_message("assistant"):
            st.error("🔑 Please enter your Groq API Key in the sidebar to run queries!")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Analyzing document..."):
                try:
                    response = query_rag_system(
                        query,
                        st.session_state.vector_db,
                        api_key=active_gemini_key,
                        groq_api_key=active_groq_key,
                        model_name=selected_model
                    )
                    answer = response["answer"]
                    context = response["context"]
                    
                    st.markdown(answer)
                    with st.expander("🔍 View Source Citations", expanded=False):
                        display_source_tabs(context)
                        
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "context": context
                    })
                except Exception as e:
                    if "429" in str(e) or "quota" in str(e).lower() or "rate_limit" in str(e).lower():
                        warning_msg = "⚠️ API rate limit reached. Please wait 30 seconds and try again!"
                        st.warning(warning_msg)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": warning_msg
                        })
                    else:
                        st.error(f"Error: {e}")