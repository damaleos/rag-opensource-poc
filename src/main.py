from fastapi import FastAPI
#from pydantic import BaseModel
from src.routers import query
from src.vector_store.loader import load_vectorstore
from src.llm.ollama_client import OllamaClient


app = FastAPI()

@app.on_event("startup")
async def startup_event():
    load_vectorstore()
    OllamaClient.load_model()


app.include_router(query.router)

'''
class PromptRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Test RAG open source - POC"}

@app.post("/query")
async def query(prompt: PromptRequest):
    user_input = prompt.text
    # Aquí iría tu lógica con LLM, por ahora retornamos un mensaje simulado
    response = f"Simulated response to: '{user_input}'"
    return {"response": response}
'''