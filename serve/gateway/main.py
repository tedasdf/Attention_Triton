import os
import httpx
from fastapi import FastAPI, Request

app = FastAPI()

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://vllm:8000")


@app.get("/")
async def root():
    return {"message": "Hello World", "vllm_url": VLLM_BASE_URL}


@app.post("/v1/chat/completions")
async def proxy_chat(request: Request):
    body = await request.json()
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{VLLM_BASE_URL}/v1/chat/completions", json=body)
    return response.json()
