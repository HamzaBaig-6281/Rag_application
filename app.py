import os
from io import BytesIO
import uuid
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# RAG dependencies
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

load_dotenv()

app = FastAPI(title="Aura RAG Web Service")

# In-memory session store for RAG contexts
# maps file_id -> FAISS vector store
vector_stores = {}

# Global Hugging Face embeddings model to avoid reloading on each request
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

class QueryRequest(BaseModel):
    query: str
    file_id: str
    api_key: Optional[str] = None

@app.post("/api/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Check that file is PDF
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        # Read the file content directly from bytes
        content = await file.read()
        reader = PdfReader(BytesIO(content))
        
        extracted_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"

        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="The uploaded PDF contains no readable text.")

        # Text Splitting
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_text(extracted_text)

        # Build FAISS index
        vector_store = FAISS.from_texts(chunks, embeddings)
        
        # Save to our session store
        file_id = str(uuid.uuid4())
        vector_stores[file_id] = vector_store

        return JSONResponse(content={
            "success": True, 
            "file_id": file_id,
            "filename": file.filename,
            "message": f"Successfully loaded and indexed {file.filename}"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
async def query_pdf(request: QueryRequest):
    try:
        file_id = request.file_id
        if file_id not in vector_stores:
            raise HTTPException(status_code=404, detail="Session not found or expired. Please upload the PDF again.")

        vector_store = vector_stores[file_id]
        
        # Determine API Key: Provided by user request or environment
        groq_api_key = request.api_key or os.environ.get("GROQ_API_KEY", "")
        if not groq_api_key:
            raise HTTPException(status_code=400, detail="Groq API Key is required. Please provide it.")

        # 1. Similarity Search
        docs = vector_store.similarity_search(request.query, k=4)
        context = "\n\n".join([doc.page_content for doc in docs])

        # 2. Construct precise system prompt
        system_prompt = (
            "You are a highly precise document QA assistant. Use the following PDF context "
            "to answer the user's question. If you don't know the answer or the context doesn't "
            "contain it, answer exactly with: 'I am sorry, but the provided PDF does not contain "
            "the answer to your question.'\nDo NOT use outside knowledge. Do NOT speculate or make up information.\n\n"
            f"Context:\n{context}"
        )

        # 3. Chat with Groq API
        chat = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            groq_api_key=groq_api_key,
            temperature=0.0
        )
        
        messages = [
            ("system", system_prompt),
            ("human", request.query)
        ]
        
        response = chat.invoke(messages)
        
        return JSONResponse(content={"answer": response.content})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Create static folder and serve the HTML frontend
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    # Render binds the application to the PORT environment variable
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
