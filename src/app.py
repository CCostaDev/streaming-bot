import os
import streamlit as st
import requests
import tempfile
import json
import hashlib
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


# --- Setup ---

load_dotenv()
st.set_page_config(page_title="RAGnet", page_icon="🤖")
st.title("RAGnet")
# st.markdown("<h1 style='margin-bottom: 0;'>Noxa</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size:16px; color:gray; font-style:italic;'>Know Fast. Act Faster.</p>", unsafe_allow_html=True)

# Vector store setup
persist_directory = "buff_chroma_db"
collection_name = "buff_docs"
embedding_model = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

vectorstore = Chroma(
    persist_directory=persist_directory,
    collection_name=collection_name,
    embedding_function=embedding_model
)
existing_docs = vectorstore.get()
st.sidebar.text(f"Chunks in DB: {len(existing_docs['documents'])}")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Helper ---
def get_file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

# --- Sidebar ---

with st.sidebar:
    st.header("📄 Document Manager")
    
    with st.expander("📚 View Uploaded Documents", expanded=False):
        existing = vectorstore.get()
        all_metadatas = existing.get("metadatas", [])

        seen_files = {}
        for meta in all_metadatas:
            if isinstance(meta, dict):
                file_hash = meta.get("file_hash")
                filename = meta.get("filename") or meta.get("source") or "Unknown"
                size = meta.get("file_size_kb", "N/A")

                # Group by file hash to avoid duplicates
                if file_hash and file_hash not in seen_files:
                    seen_files[file_hash] = {
                        "filename": filename,
                        "size": size
                    }

        if not seen_files:
            st.markdown("_No documents currently in the database._")
        else:
            for file_hash, info in seen_files.items():
                col1, col2 = st.columns([0.75, 0.25])
                with col1:
                    st.markdown(f"**{info['filename']}** – {info['size']} KB")
                with col2:
                    if st.button("🗑️", key=f"delete_{file_hash}"):
                        existing = vectorstore.get()
                        all_ids = existing.get("ids", [])
                        all_metadatas = existing.get("metadatas", [])

                        ids_to_delete = [
                            doc_id
                            for doc_id, meta in zip(all_ids, all_metadatas)
                            if isinstance(meta, dict) and meta.get("file_hash") == file_hash
                        ]

                        if ids_to_delete:
                            vectorstore.delete(ids=ids_to_delete)
                            st.rerun()


    uploaded_file = st.file_uploader("Upload PDF or CSV", type=["pdf", "csv"])
    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        file_hash = get_file_hash(file_bytes)

        # Check if the has is alreay in vectorstore metadata
        existing = vectorstore.get()
        all_metadatas = existing.get("metadatas", [])
        flat_hashes = {
            meta.get("file_hash")
            for meta in all_metadatas if isinstance(meta, dict)
        }

        if file_hash in flat_hashes:
            st.info("This file is already in the database.")
        else:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(uploaded_file.getvalue())
                file_path = tmp.name

            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                source_type = "pdf"
            elif uploaded_file.name.endswith(".csv"):
                loader = CSVLoader(file_path)
                source_type = "csv"

            documents = loader.load()
            chunks = text_splitter.split_documents(documents)

            for doc in chunks:
                doc.metadata["source_type"] = source_type
                doc.metadata["filename"] = uploaded_file.name
                doc.metadata["file_hash"] = file_hash
                doc.metadata["file_size_kb"] = round(len(file_bytes) / 1024, 2)
        
            vectorstore.add_documents(chunks)
            st.success(f"{uploaded_file.name} added to vector database ✅")
            os.remove(file_path)


    # API dummy data to be replaced with API later
    if st.button("📡 Add Example API Data"):
        api_data = requests.get("https://jsonplaceholder.typicode.com/users").json()
        pretty_json = json.dumps(api_data, indent=2)
        file_hash = get_file_hash(pretty_json.encode())

        # Check if already added
        existing = vectorstore.get()

        all_metadatas = existing.get("metadatas", [])
        flat_hashes = {
            meta.get("file_hash")
            for meta in all_metadatas if isinstance(meta, dict)
        }

        if file_hash in flat_hashes:
            st.info("API data already added.")
        else:
            doc = Document(
                page_content=pretty_json,
                metadata={
                    "source_type": "api",
                    "source": "jsonplaceholder",
                    "file_hash": file_hash
                }
            )
            vectorstore.add_documents([doc])
            st.success("API data added to vector database ✅")
            st.text_area("📄 Preview API data added", pretty_json, height=300)

    if st.button("🧹 Clear Vector Database"):
        vectorstore.delete_collection()
        st.success("Database cleared")

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.success("Chat history cleared")


# --- Chat logic ---

# Get response
def get_response(query, chat_history):
    retriever = vectorstore.as_retriever()
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    template = """
        You are a helpful assistant. Answer the following questions considering the history of the conversation:

        Context: {context}

        Chat history: {chat_history}

        User question: {user_question}
        """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatOpenAI(model="gpt-4.1-mini")  # "gpt-4o" for better results, "gpt-4o-mini" for cheaper and quick results (moderate reasoning), "gpt-4.1-mini" for quicker results but more expensive that 4o-mini (moderate reasoning)

    chain = prompt | llm | StrOutputParser()

    return chain.stream({
        "context": context,
        "chat_history": chat_history,
        "user_question": query
    })
    
# --- Chat UI ---

# Conversation
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)


# User input
user_query = st.chat_input("Your message")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        ai_placeholder = st.empty()
        ai_placeholder.markdown("*Thinking...*") 

        response_stream = get_response(user_query, st.session_state.chat_history)
        full_response = ""
        first_chunk = None

        try:
            first_chunk = next(response_stream)
        except StopIteration:
            first_chunk = ""

        ai_placeholder.markdown(first_chunk)
        ai_response = ai_placeholder.write_stream(response_stream)
        full_response = first_chunk + ai_response

    st.session_state.chat_history.append(AIMessage(full_response))

