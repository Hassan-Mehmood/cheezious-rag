from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.rag import router as rag_router

from src.logger import logger

app = FastAPI()
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
    logger.info("TESTING")
    return {"message": "Hello World"}

@app.get("/health")
async def health():
    logger.info('HEALTH')
    return {'status': 'ok'}