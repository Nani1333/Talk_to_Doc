import os
from dotenv import load_dotenv
import chromadb
import google.generativeai as genai
from chromadb.utils import embedding_functions

# Load environment variables from .env file
load_dotenv()

# Get the Google API Key from environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

# Configure the Google Generative AI client
genai.configure(api_key=google_api_key)

# Initialize Google's embedding function for ChromaDB
google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    api_key=google_api_key,
    model_name="models/embedding-001"  # Embedding model
)

# Setup ChromaDB with persistent storage
chroma_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")  # default path fallback
chroma_client = chromadb.PersistentClient(path=chroma_path)

# Get or create a collection with embedding function
collection = chroma_client.get_or_create_collection(
    name="document_qa_collection",
    embedding_function=google_ef
)

# Initialize Gemini chat model
chat_model = genai.GenerativeModel("gemini-1.5-flash")

# Load documents from a directory
def load_documents_from_directory(directory_path):
    print(f"Loading documents from directory: {directory_path}")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(os.path.join(directory_path, filename), "r", encoding="utf-8") as file:
                documents.append({"id": filename, "text": file.read()})
    return documents

# Split text into overlapping chunks
def split_text_into_chunks(text, chunk_size=1000, overlap=50):
    print(f"Splitting text into chunks of size {chunk_size}...")
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

# Function to generate embeddings using Google
def get_google_embedding(text):
    print("==== Generating embeddings with Google Generative AI... ====")
    embedding = google_ef([text])[0]
    return embedding

# Load and chunk documents
directory_path = "./news_articles"
documents = load_documents_from_directory(directory_path)
print(f"Loaded {len(documents)} documents from {directory_path}")

# Chunk documents
chunked_documents = []
for doc in documents:
    chunks = split_text_into_chunks(doc["text"])
    print(f"Splitting document {doc['id']} into {len(chunks)} chunks.")
    for i, chunk in enumerate(chunks):
        chunked_documents.append({
            "id": f"{doc['id']}_chunk_{i}",
            "text": chunk
        })

# Generate embeddings for all chunks
for doc in chunked_documents:
    doc["embedding"] = get_google_embedding(doc["text"])

# Upsert chunked documents into ChromaDB
print("Upserting chunked documents into ChromaDB collection...")
for doc in chunked_documents:
    collection.upsert(
        ids=[doc["id"]],
        documents=[doc["text"]],
        embeddings=[doc["embedding"]]
    )

# Function to query similar chunks
def query_documents(question, n_results=5):
    print(f"\n==== Querying for: \"{question}\" ====")
    results = collection.query(query_texts=[question], n_results=n_results)
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    print("==== Retrieved relevant chunks ====")
    return relevant_chunks

# Function to generate response using Gemini
def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )
    try:
        response = chat_model.generate_content(
            contents=[{"role": "user", "parts": prompt}]
        )
        return response.text
    except Exception as e:
        print(f"Gemini API error: {e}")
        return "Sorry, I couldn't generate an answer right now."

# Example usage
question = "Tell me about the SpaceX Starship launch."
relevant_chunks = query_documents(question)
answer = generate_response(question, relevant_chunks)

print("\n==== Final Answer ====")
print(answer)
