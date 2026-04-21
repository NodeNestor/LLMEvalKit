#!/usr/bin/env python
"""Template: wrap a custom PyTorch model as an OpenAI-compatible endpoint.

Copy this file into your project, fill in MODEL_LOAD and GENERATE, then
reference it from a model YAML:

    # configs/my_custom.yaml
    name: my-custom
    kind: custom
    script: /path/to/your/project/serve.py
    model_name: my-custom-v1
    port: 8001
    env:
        CUDA_VISIBLE_DEVICES: "1"

Run:
    python -m llmevalkit run --model configs/my_custom.yaml --profile quick

Design notes
------------
- Supports /v1/chat/completions (non-streaming) and /v1/models.
- Drops streaming and tool_calls for simplicity; most eval harnesses
  don't need them. Add later if required.
- You own the chat template: build the prompt from the messages list.
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

# ---------------------------------------------------------------------------
# 1. Load your model here.
# ---------------------------------------------------------------------------

MODEL = None
TOKENIZER = None
MODEL_NAME = "my-custom-model-v1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model() -> None:
    """Fill in — load weights, set up whatever routing etc. you need."""
    global MODEL, TOKENIZER
    # Example: HF transformers
    # from transformers import AutoModelForCausalLM, AutoTokenizer
    # MODEL = AutoModelForCausalLM.from_pretrained(
    #     "LiquidAI/LFM2.5-350M-instruct", torch_dtype=torch.bfloat16, device_map=DEVICE
    # )
    # TOKENIZER = AutoTokenizer.from_pretrained("LiquidAI/LFM2.5-350M-instruct")
    # MODEL.eval()
    raise NotImplementedError("implement load_model() for your custom model")


def generate(messages: list[dict], max_tokens: int, temperature: float, **kw) -> str:
    """Fill in — messages-in, reply-string-out.

    Run your custom forward here (routed execution, 1-bit weights, ES-trained
    weights, or any novel architecture).
    """
    # Example with transformers:
    # prompt = TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # inputs = TOKENIZER(prompt, return_tensors="pt").to(DEVICE)
    # with torch.inference_mode():
    #     out = MODEL.generate(
    #         **inputs, max_new_tokens=max_tokens, temperature=max(temperature, 1e-5),
    #         do_sample=temperature > 0,
    #     )
    # new_tokens = out[0][inputs.input_ids.shape[1]:]
    # return TOKENIZER.decode(new_tokens, skip_special_tokens=True)
    raise NotImplementedError("implement generate() for your custom model")


# ---------------------------------------------------------------------------
# 2. OpenAI-compatible HTTP surface — usually no need to edit below.
# ---------------------------------------------------------------------------

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
    return {
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "local"}],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    messages = [m.model_dump() for m in req.messages]
    reply = generate(
        messages, max_tokens=req.max_tokens or 512,
        temperature=req.temperature or 0.7,
        top_p=req.top_p or 1.0, stop=req.stop,
    )
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": reply},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


@app.get("/health")
def health():
    return {"ok": MODEL is not None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--model-name", default=None)
    args = ap.parse_args()

    global MODEL_NAME
    if args.model_name:
        MODEL_NAME = args.model_name

    load_model()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
