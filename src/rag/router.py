from fastapi import APIRouter, HTTPException, Depends, File, UploadFile
from typing import Dict, Any, List
from pydantic import BaseModel
from src.rag.Rag import Rag
from src.logger import logger
from langchain.document_loaders import TextLoader
from io import StringIO
import os

router = APIRouter(prefix="/rag", tags=["rag"])

# Pydantic models for request/response validation
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class QueryResponse(BaseModel):
    query: str
    response: str
    sources: List[Dict[str, Any]]

# Dependency injection for RAG service
def get_rag_service():
    try:
        return Rag()
    except Exception as e:
        logger.error(f"Failed to initialize RAG service: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to initialize RAG service"
        )

@router.post("/query", response_model=QueryResponse)
async def query_rag(
    request: QueryRequest,
    rag: Rag = Depends(get_rag_service)
) -> Dict[str, Any]:
    """
    Process a query using the RAG system
    """
    try:
        logger.info(f"Received query request: {request.query}")
        result = rag.query(request.query, request.top_k)
        return result
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to process query"
        )

@router.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    rag: Rag = Depends(get_rag_service)
) -> Dict[str, Any]:
    """
    Ingest a document into the RAG system
    """
    try:
        logger.info(f"Received document for ingestion: {file.filename}")
        
        content = await file.read()
        text_content = content.decode("utf-8")
        
        # Create a temporary file
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "w", encoding="utf-8") as f:
            f.write(text_content)
        
        # Load the document
        loader = TextLoader(temp_file_path)
        documents = loader.load()
        
        # Add metadata
        for doc in documents:
            doc.metadata["source"] = file.filename
        
        # Store in RAG
        success = rag.store(documents)
        
        # Clean up
        os.remove(temp_file_path)
        
        if success:
            return {
                "status": "success",
                "message": f"Document '{file.filename}' ingested successfully"
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to ingest document"
            )
            
    except Exception as e:
        logger.error(f"Error ingesting document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to ingest document: {str(e)}"
        )

@router.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint for the RAG system
    """
    try:
        # Just initialize the RAG service to check if it's working
        rag = get_rag_service()
        return {"status": "healthy"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="RAG system is not healthy"
        )
