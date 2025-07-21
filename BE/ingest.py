import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.embeddings import HuggingFaceEmbeddings

# --- Configuration ---
SOURCE_DOCS_PATH = "data"
VECTOR_STORE_PATH = "vector_store.faiss"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# A mapping from file extensions to document loaders
DOCUMENT_MAP = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
    ".txt": TextLoader,
}

def main():
    """
    Main function to ingest documents, create embeddings, and build a vector store.
    """
    print("Starting document ingestion process...")

    if not os.path.exists(SOURCE_DOCS_PATH):
        print(f"Error: Source documents directory not found at '{SOURCE_DOCS_PATH}'.")
        print("Please create this directory and add your PDF, DOCX, or TXT files.")
        return

    # --- 1. Load Documents ---
    all_docs = []
    for filename in os.listdir(SOURCE_DOCS_PATH):
        file_path = os.path.join(SOURCE_DOCS_PATH, filename)
        ext = "." + filename.rsplit(".", 1)[-1]
        if ext in DOCUMENT_MAP:
            loader_class = DOCUMENT_MAP[ext]
            print(f"Loading {filename}...")
            loader = loader_class(file_path)
            all_docs.extend(loader.load())
        else:
            print(f"Skipping {filename}: unsupported file type.")

    if not all_docs:
        print("No documents loaded. Exiting.")
        return

    print(f"Loaded {len(all_docs)} document pages/sections.")

    # --- 2. Split Documents into Chunks ---
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_texts = text_splitter.split_documents(all_docs)
    print(f"Split documents into {len(split_texts)} chunks.")

    # --- 3. Create Embeddings ---
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'} # Use CPU for broad compatibility
    )

    # --- 4. Build and Save Vector Store ---
    print("Creating vector store from documents...")
    vector_store = FAISS.from_documents(split_texts, embeddings)
    
    print(f"Saving vector store to {VECTOR_STORE_PATH}...")
    vector_store.save_local(VECTOR_STORE_PATH)
    
    print("\nIngestion complete!")
    print(f"Vector store saved locally at '{VECTOR_STORE_PATH}'.")
    print(f"You can now run the main API application.")

if __name__ == "__main__":
    main()
