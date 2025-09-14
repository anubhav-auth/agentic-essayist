import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Define the path for the Chroma vector store persistence
CHROMA_PATH = "chroma"
DATA_PATH = "data"

def main():
    print("Starting data ingestion process...")
    documents = load_documents()
    chunks = split_documents(documents)
    save_to_chroma(chunks)
    print("Data ingestion complete.")

def load_documents():
    print(f"Loading documents from {DATA_PATH}...")
    loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader) # [cite: 210]
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")
    return documents

def split_documents(documents):
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # [cite: 211]
        chunk_overlap=100, # [cite: 211]
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks):
    print("Loading embedding model and saving to ChromaDB...")
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        import shutil
        shutil.rmtree(CHROMA_PATH)

    # Load the specified embedding model from Hugging Face. [cite: 213]
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create a new ChromaDB database from the documents. [cite: 214]
    db = Chroma.from_documents(
        chunks, embeddings, persist_directory=CHROMA_PATH
    )
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    main()