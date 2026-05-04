# Aura RAG Desktop Assistant

A premium, simple, standalone desktop interface for a complete PDF Retrieval-Augmented Generation (RAG) system. No backend, Streamlit, or complicated configurations required. 

## Features
- **Upload PDF**: Process and embed directly via local `HuggingFaceEmbeddings` and search using a local FAISS index.
- **Advanced Contextual Answering**: Uses LangChain and `ChatGroq` (`llama-3.3-70b-versatile`) to strictly answer questions based on the PDF content.
- **Intuitive GUI**: Built using Python's native `Tkinter` framework.

## How to Run
1. Install Python 3.x.
2. Install the required dependencies:
   ```bash
   pip install pypdf langchain-text-splitters langchain-huggingface langchain-community faiss-cpu langchain-groq python-dotenv
   ```
3. Create a `.env` file containing your `GROQ_API_KEY`:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```
4. Run the application:
   ```bash
   python app.py
   ```
