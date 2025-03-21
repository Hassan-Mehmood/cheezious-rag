from fastapi import APIRouter

router = APIRouter(prefix="/rag", tags=["rag"])

@router.get("/")
async def root():
    pass
