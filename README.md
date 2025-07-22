# 🔍 RAG_basic — Retrieval-Augmented Generation using ChromaDB + Gemini

**RAG_basic** is a simple implementation of a Retrieval-Augmented Generation (RAG) pipeline using:

- 🧠 **Google Gemini 1.5 Flash** for natural language understanding and response generation
- 🗃️ **ChromaDB** for fast vector similarity search
- 🧬 **Google GenerativeAI Embeddings** for semantic indexing
- 📄 **Local `.txt` documents** for knowledge base

---

## 📌 Features

- Load `.txt` documents from a folder
- Split them into overlapping text chunks
- Generate vector embeddings via Google GenerativeAI
- Store and retrieve them using ChromaDB
- Use Gemini to answer natural language queries based on the most relevant chunks

---

## 📂 Project Structure

```

RAG\_basic/
├── app.py                # Main application script
├── .env                  # Environment variables (API key, DB path)
├── chroma\_db/            # Local vector database (auto-created)
└── news\_articles/        # Folder for input .txt documents

````

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Nani1333/RAG_basic.git
cd RAG_basic
````

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary>📦 Required packages</summary>

You can add this in `requirements.txt`:

```
python-dotenv
chromadb
google-generativeai
```

</details>

---

### 4. Setup `.env` file

Create a `.env` file in the root directory:

```
GOOGLE_API_KEY=your_google_api_key
CHROMA_DB_PATH=./chroma_db
```

> Get your Gemini API key from: [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)

---

### 5. Add your documents

Place your `.txt` files inside the `news_articles/` folder.
Each file should contain plain text content that can be used for answering questions.

---

## 🚀 Run the Script

```bash
python app.py
```

> Example question in `app.py`:
>
> ```
> question = "Tell me about Databricks"
> ```

If any documents contain relevant information, Gemini will respond accordingly. Otherwise, you'll get a fallback like:

> *"I'm sorry, but this document does not contain any information about Databricks."*

---

## 🧠 How it works

1. Loads `.txt` files
2. Splits each into chunks (for semantic retrieval)
3. Embeds each chunk using Google's embedding API
4. Stores them into ChromaDB
5. On user query:

   * Finds top relevant chunks
   * Sends them + the question to Gemini
   * Returns an answer

---

## ✅ Example Output

```
==== Querying for: "Tell me about Databricks" ====
==== Retrieved relevant chunks ====
==== Final Answer ====
Databricks is a platform that provides tools for building and managing big data and AI applications...
```

---

## 🧩 Future Improvements

* Add support for PDF files (`PyMuPDF`)
* Use batch embeddings for performance
* Enable web scraping or document previews
* Web UI using Streamlit or Flask

---

## 🙋‍♂️ Author

Developed by [T. Sriram](https://github.com/Nani1333)

Feel free to open issues or suggestions!
