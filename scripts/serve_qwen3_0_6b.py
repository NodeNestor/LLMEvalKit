#!/usr/bin/env python
"""Test shim: serves Qwen3-0.6B via transformers as an OpenAI-compatible endpoint.

Used to sanity-check LLMEvalKit end-to-end without heavy inference deps.
Copied from custom_model_server_template.py and filled in for Qwen3-0.6B.
"""
from __future__ import annotations

import argparse
import time
import uuid
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = None
TOKENIZER = None
MODEL_NAME = "qwen3-0.6b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model() -> None:
    global MODEL, TOKENIZER
    model_id = "Qwen/Qwen3-0.6B"
    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    TOKENIZER = AutoTokenizer.from_pretrained(model_id)
    MODEL = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(DEVICE)
    MODEL.eval()
    print(f"[shim] loaded {model_id} on {DEVICE}, dtype={dtype}")


def generate(messages: list[dict], max_tokens: int, temperature: float,
             top_p: float = 1.0, stop: Optional[list[str]] = None, **_) -> str:
    prompt = TOKENIZER.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = TOKENIZER(prompt, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        out = MODEL.generate(
            **inputs, max_new_tokens=max_tokens,
            do_sample=temperature > 1e-4,
            temperature=max(temperature, 1e-4),
            top_p=top_p,
            pad_token_id=TOKENIZER.eos_token_id,
        )
    new_tokens = out[0][inputs.input_ids.shape[1]:]
    return TOKENIZER.decode(new_tokens, skip_special_tokens=True).strip()


app = FastAPI()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False
    stop: Optional[list[str]] = None


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    reply = generate(
        [m.model_dump() for m in req.messages],
        max_tokens=req.max_tokens or 256,
        temperature=req.temperature or 0.7,
        top_p=req.top_p or 1.0, stop=req.stop,
    )
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": reply},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


@app.get("/health")
def health():
    return {"ok": MODEL is not None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()
    load_model()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
