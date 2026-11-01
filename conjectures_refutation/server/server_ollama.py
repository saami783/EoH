from __future__ import annotations

import os
from typing import Any
import requests

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Ollama EoH Proxy")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")


class LLMRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    repeat_prompt: int | None = 1
    params: dict[str, Any] | None = None
    temperature: float | None = 0.8
    model: str | None = None


class LLMResponse(BaseModel):
    content: list[str]
    success: bool = True
    response: str = ""
    model: str | None = None


@app.get("/health")
def health():
    try:
        res = requests.get("http://localhost:11434/", timeout=2)
        ollama_ok = res.status_code == 200
        return {
            "status": "ok",
            "ollama_reachable": ollama_ok,
            "model_configured": OLLAMA_MODEL,
        }
    except Exception as e:
        return {"status": "degraded", "ollama_reachable": False, "detail": str(e)}


@app.post("/api/chat/ollama", response_model=LLMResponse)
def chat_eoh(req: LLMRequest):
    temp = req.temperature
    if req.params and req.params.get("temperature") is not None:
        temp = req.params["temperature"]
    if temp is None:
        temp = 0.8

    print(f"REQ - Prompt length: {len(req.prompt)} | Temp: {temp}")

    model_to_use = req.model if req.model else OLLAMA_MODEL

    payload = {
        "model": model_to_use,
        "prompt": req.prompt,
        "stream": False,
        "options": {
            "temperature": temp,
        },
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=300.0)
        response.raise_for_status()

        data = response.json()
        raw_output = data.get("response", "")

        print("RAW OUTPUT:", repr(raw_output[:200] + "..." if len(raw_output) > 200 else raw_output))

        return LLMResponse(
            content=[raw_output],
            success=True,
            response=raw_output,
            model=model_to_use,
        )

    except requests.exceptions.RequestException as e:
        print(f"[Ollama Proxy] Erreur de communication avec Ollama: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "Erreur de communication avec Ollama", "exception": str(e)},
        ) from e
    except Exception as e:
        print(f"[Ollama Proxy] Erreur interne: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "Erreur interne du serveur proxy", "exception": str(e)},
        ) from e