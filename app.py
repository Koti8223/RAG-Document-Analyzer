import os
import streamlit as st
import tempfile
from dotenv import load_dotenv

# Import our backend functions
from rag_pipeline import get_document_chunks, initialize_vector_db, query_rag_system
from langchain_google_genai import GoogleGenerativeAIEmbeddings
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

def display_source_tabs(context):
    """
    Renders source citations inside a clean, modern tabbed interface.
    """
    if not context:
        return
        
    st.markdown("### 🔍 Source Citations")
    tabs = st.tabs([f"📄 Source {i+1} (Page {doc.metadata.get('page', 'N/A')})" for i, doc in enumerate(context)])
    
    for idx, tab in enumerate(tabs):
        doc = context[idx]
        with tab:
            filename = os.path.basename(doc.metadata.get("source", "Unknown File"))
            page_num = doc.metadata.get("page", "N/A")
            st.caption(f"📂 **File:** `{filename}` | 📄 **Page:** `{page_num}`")
            formatted_text = "\n> ".join(doc.page_content.split("\n"))
            st.markdown(f"> {formatted_text}")

# --- SIDEBAR (For uploading and settings) ---
with st.sidebar:
    st.title("⚙️ Document Settings")
    
    # Check if a default system key is configured in your env/secrets
    system_key = os.getenv("GOOGLE_API_KEY")
    system_key_exists = system_key is not None and system_key.strip() != ""
    
    # Dynamically change label: Optional locally, Required for public users
    key_label = "Gemini API Key (Optional)" if system_key_exists else "Gemini API Key (Required)"
    placeholder_text = "Uses local system key by default..." if system_key_exists else "Required to run queries..."
    
    # Security: Bring Your Own Key (BYOK) Input Box
    user_key = st.text_input(
        key_label, 
        type="password", 
        placeholder=placeholder_text,
        help="Get a free key from Google AI Studio. Since this is a public demo, providing your own key is required." if not system_key_exists else "If left blank, it will use the default system key."
    )
    
    # Determine which key to use (User key takes priority over default key)
    active_key = user_key if user_key else os.getenv("GOOGLE_API_KEY")
    
    st.write("---")
    
    uploaded_file = st.file_uploader(
        "Upload a Research Paper (PDF)", 
        type=["pdf"], 
        help="Upload any PDF to parse text, tables, and images."
    )
    
    # CASE A: A file is actively uploaded
    if uploaded_file is not None:
        if not active_key:
            st.error("🔑 Please enter a Gemini API Key or configure the system key in Secrets!")
        else:
            if uploaded_file.name != st.session_state.current_file:
                st.session_state.current_file = uploaded_file.name
                st.session_state.chat_history = []  # Clear history for new file
                
                db_dir = f"./chroma_db_{uploaded_file.name}"
                
                with st.status("Reading and indexing document...", expanded=True) as status:
                    try:
                        # Performance: Check if the vector database already exists on disk (Caching)
                        if os.path.exists(db_dir):
                            status.write("⚡ Found existing database on disk. Loading instantly...")
                            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2", google_api_key=active_key)
                            st.session_state.vector_db = Chroma(
                                persist_directory=db_dir,
                                embedding_function=embeddings
                            )
                            status.update(label="Database loaded from cache!", state="complete", expanded=False)
                            st.success("Loaded from disk instantly! Ready for Q&A.")
                        else:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                tmp_file_path = tmp_file.name
                            
                            status.write("Parsing text and tables locally...")
                            chunks = get_document_chunks(tmp_file_path, api_key=active_key)
                            
                            status.write("Generating embeddings and building vector database...")
                            st.session_state.vector_db = initialize_vector_db(chunks, api_key=active_key, persist_directory=db_dir)
                            
                            os.remove(tmp_file_path)
                            status.update(label="Document indexed successfully!", state="complete", expanded=False)
                            st.success("Ready for Q&A!")
                    except Exception as e:
                        status.update(label="Indexing failed!", state="error")
                        st.error(f"Error: {e}")
                        st.session_state.vector_db = None
                        st.session_state.current_file = None
                    
    # CASE B: No file is uploaded
    else:
        if st.session_state.current_file is not None:
            st.session_state.current_file = None
            st.session_state.vector_db = None
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
            st.warning("Please upload a PDF document in the sidebar first!")
    elif not active_key:
        with st.chat_message("assistant"):
            st.error("🔑 Please enter a Gemini API Key to run queries!")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Analyzing document..."):
                try:
                    response = query_rag_system(query, st.session_state.vector_db, api_key=active_key)
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
                    if "429" in str(e) or "quota" in str(e).lower():
                        warning_msg = "⚠️ Gemini API is temporarily rate-limited. Please wait 30 seconds and try again!"
                        st.warning(warning_msg)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": warning_msg
                        })
                    else:
                        st.error(f"Error: {e}")