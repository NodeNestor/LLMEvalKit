#!/usr/bin/env python
"""Test shim: serves LFM2-1.2B via transformers as OpenAI-compatible endpoint."""
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
MODEL_NAME = "lfm2-1.2b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model() -> None:
    global MODEL, TOKENIZER
    model_id = "LiquidAI/LFM2-1.2B"
    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    TOKENIZER = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    MODEL = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True,
    ).to(DEVICE)
    MODEL.eval()
    print(f"[shim] loaded {model_id} on {DEVICE}, dtype={dtype}", flush=True)


def generate(messages: list[dict], max_tokens: int, temperature: float,
             top_p: float = 1.0, stop: Optional[list[str]] = None, **_) -> str:
    try:
        prompt = TOKENIZER.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = TOKENIZER.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
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
    ap.add_argument("--port", type=int, default=8003)
    args = ap.parse_args()
    load_model()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
