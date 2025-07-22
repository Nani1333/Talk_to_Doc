import os
from dotenv import load_dotenv
import chromadb
import google.generativeai as genai
from chromadb.utils import embedding_functions

# Load environment variables from .env file (e.g., GOOGLE_API_KEY)
load_dotenv()

# Get the Google API Key from environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

# Configure the Google Generative AI client with your API key
genai.configure(api_key=google_api_key)

# Initialize Google's embedding function for ChromaDB
google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    api_key=google_api_key,
    model_name="models/embedding-004" # Specifies the embedding model
)

# Initialize ChromaDB client with persistent storage and get/create a collection
chroma_client = chromadb.PersistentClient(path="CHROMA_DB_PATH")
collection = chroma_client.get_or_create_collection(
    name="document_qa_collection",
    embedding_function=google_ef
)

# Initialize the Gemini chat model (e.g., 'gemini-1.5-flash')
chat_model = genai.GenerativeModel('gemini-1.5-flash')

# Make a chat completion request to the Gemini model
try:
    response = chat_model.generate_content(
        contents=[
            {"role": "user", "parts": "What is human life expectancy in the United States?"}
        ]
    )
    # Print the text content from the Gemini model's response
    print(response.text)

except Exception as e:
    # Handle any errors during the API call
    print(f"An error occurred during Gemini API call: {e}")