# from src import app
import uvicorn

if __name__ == "__main__":
    uvicorn.run("src:app", host="localhost", port=8000, reload=True)
