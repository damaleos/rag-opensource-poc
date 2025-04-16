from fastapi import APIRouter
from pydantic import BaseModel
from src.llm.ollama_client import OllamaClient

class PromptRequest(BaseModel):
    text: str

router = APIRouter()

@router.post("/query")
async def query(prompt: PromptRequest):
    user_input = prompt.text
    llmClient = OllamaClient()
    response = llmClient.invoke(user_input)
    #response = f"query.py - Simulated response to: '{user_input}'"
    return {"response": response}