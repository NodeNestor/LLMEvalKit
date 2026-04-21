"""Simple chat sanity runner — no external framework, no download.

Sends a handful of hand-crafted prompts through /v1/chat/completions and
writes raw replies. Use this to confirm your adapter is alive before
kicking off anything heavy.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import httpx

from .base import Runner, RunResult
from ..adapters.base import Endpoint

DEFAULT_PROMPTS = [
    ("hello", "Reply with exactly: pong"),
    ("math", "What is 137 * 42? Reply with only the number."),
    ("reason", "Alice is older than Bob. Bob is older than Carol. Who is youngest? One word."),
    ("code", "Write a Python one-liner that returns the nth Fibonacci number. Output only code, no prose."),
    ("refuse", "Reply in three words with your safest refusal of a harmful request."),
    ("ifeval", "List exactly three European capital cities, separated by commas, lowercase."),
]


class SimpleChat(Runner):
    name = "simple-chat"
    requires: list[str] = []

    def run(self, endpoint: Endpoint, tasks: list[str], output_dir: Path,
            prompts: list[tuple[str, str]] | None = None,
            max_tokens: int = 256, temperature: float = 0.2,
            timeout: float = 120.0, **kwargs) -> RunResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        prompts = prompts or DEFAULT_PROMPTS

        headers = {"Authorization": f"Bearer {endpoint.api_key}",
                   "Content-Type": "application/json"}
        if endpoint.extra_headers:
            headers.update(endpoint.extra_headers)

        results: list[dict] = []
        t0 = time.time()
        success = True
        err: str | None = None
        with httpx.Client(timeout=timeout) as client:
            for tag, prompt in prompts:
                payload = {
                    "model": endpoint.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                try:
                    r = client.post(
                        f"{endpoint.base_url}/chat/completions",
                        headers=headers, json=payload,
                    )
                    r.raise_for_status()
                    data = r.json()
                    reply = data["choices"][0]["message"]["content"]
                    results.append({"tag": tag, "prompt": prompt, "reply": reply})
                except Exception as e:
                    success = False
                    err = f"{tag}: {e}"
                    results.append({"tag": tag, "prompt": prompt, "error": str(e)})

        (output_dir / "replies.json").write_text(json.dumps(results, indent=2))
        scores = {"prompts_ok": sum(1 for r in results if "reply" in r),
                  "prompts_total": len(results)}

        return RunResult(
            runner=self.name, tasks=[t for t, _ in prompts], scores=scores,
            duration_s=time.time() - t0, success=success, error=err,
            raw_output_dir=str(output_dir),
        )
