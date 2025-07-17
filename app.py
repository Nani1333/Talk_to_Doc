import os
from dotenv import load_env
import chromadb 
from openai import OpenAI
from chromadb.utils import embedding_functions

#Load Environment Variables from .env file
load_env()

openai_key = os.getenv("OPENAI_API_KEY")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key = openai_key, model_name = "text-embedding-3-small"
)

