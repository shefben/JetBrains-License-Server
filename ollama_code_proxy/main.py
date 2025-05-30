from fastapi import FastAPI
from proxy_server.routes import router as ollama_router

app = FastAPI()

app.include_router(ollama_router, prefix="/api/v1/ollama", tags=["ollama"])

@app.get("/")
async def root():
    return {"message": "Ollama Code Proxy is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
