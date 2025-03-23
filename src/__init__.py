from langchain.document_loaders import PyMuPDFLoader, WebBaseLoader, TextLoader
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from src.rag_router import router as rag_router
from langchain.schema import Document
from src.rag.rag import Rag 
from typing import List
import tempfile
import shutil


app = FastAPI()
rag = Rag()

app.include_router(rag_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/add_document/")
async def add_document(file: UploadFile = File(None), url: str = Form(None)):
    """
    Endpoint to add documents in Pinecone vector database.
    Supports PDFs, text files, and website URLs.
    """
    try:
        documents: List[Document] = []

        if file:
            # Save file temporarily
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                shutil.copyfileobj(file.file, tmp_file)
                tmp_file_path = tmp_file.name

            # Detect file type and load content
            if file.filename.endswith(".pdf"):
                loader = PyMuPDFLoader(tmp_file_path)
            elif file.filename.endswith(".txt"):
                loader = TextLoader(tmp_file_path)
            else:
                return {"success": False, "message": "Unsupported file format"}

            documents = loader.load()

        elif url:
            # Load content from URL
            loader = WebBaseLoader(url)
            documents = loader.load()

        else:
            return {"success": False, "message": "No valid input provided"}

        # Store documents in vector database
        success = rag.store(documents)

        if success:
            return {
                "success": True,
                "message": "Documents added to Pinecone vector store",
            }
        else:
            return {"success": False, "message": "Failed to store documents"}

    except Exception as e:
        return {"success": False, "message": str(e)}
