import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import rag_service  # Import our modular RAG service

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Doc-RAG API",
    description="An API for asking questions to your documents.",
    version="1.0.0",
)

# --- CORS Middleware ---
# This allows our React frontend (running on a different port) to communicate with this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, you should restrict this to your frontend's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- In-Memory Storage ---
# A simple dictionary to act as an in-memory "database" for our vector stores.
# Key: filename, Value: Chroma vector store object
# This avoids re-processing the same file multiple times.
vector_store_cache = {}
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# --- API Routes ---

@app.post("/upload-and-process/")
async def upload_and_process_file(file: UploadFile = File(...)):
    """
    Handles file uploads.
    1. Saves the uploaded file to a temporary directory.
    2. Creates a vector store from the file's content.
    3. Caches the vector store in memory.
    """
    if file.filename in vector_store_cache:
        return {"status": "success", "message": f"File '{file.filename}' already processed and cached."}

    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Use our RAG service to create the vector store
        print(f"API: Creating vector store for {file.filename}...")
        vector_store = rag_service.create_vector_store(file_path)
        
        # Cache the vector store
        vector_store_cache[file.filename] = vector_store
        print(f"API: Vector store for {file.filename} cached.")
        
        return {"status": "success", "filename": file.filename, "message": "File processed and ready for questions."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


@app.post("/ask/")
async def ask_question(filename: str = Form(...), question: str = Form(...)):
    """
    Answers a question based on a previously uploaded file.
    1. Retrieves the vector store from the cache.
    2. Creates a RAG chain.
    3. Invokes the chain with the question and returns the answer.
    """
    print(f"API: Received question for {filename}: '{question}'")
    
    # Check if the vector store is in our cache
    if filename not in vector_store_cache:
        raise HTTPException(status_code=404, detail="File not found. Please upload the file first.")
    
    try:
        vector_store = vector_store_cache[filename]
        
        # Use our RAG service to create the chain and get an answer
        rag_chain = rag_service.create_rag_chain(vector_store)
        answer = rag_chain.invoke(question)
        
        return {"question": question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error asking question: {str(e)}")
