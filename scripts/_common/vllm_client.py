"""
scripts/_common/vllm_client.py
------------------------------
Thin client for the vLLM OpenAI-compatible server used by the INFERENCE steps
(teacher tweet generation, semantic-filter judging, baseline/eval). It supports
vLLM's **dynamic LoRA adapter loading**, which is the whole reason we use vLLM
instead of Ollama: the teacher = base model + a freshly-trained LoRA adapter, and
vLLM can serve the base once and hot-load the adapter by path — no merge, no GGUF
conversion.

WHAT USES THIS (inference only):
  trait folders:  01_verify_and_baseline.py, 02_generate_tweets.py,
                  03_semantic_filter.py, 05_evaluate_student.py (+ _crossmodels)
  wbTiramisu:     06_evaluate_student_approach_c.py
WHAT MUST NOT USE THIS (needs real weights/gradients -> stay pure HF):
  00_finetune_teacher.py, 04_finetune_student.py, 05_train_student_approach_c.py,
  02_compute_trait_gradient_approach_c.py, 03_score_candidates_approach_c.py

Import (PYTHONPATH=/app/scripts is set in the container):
    from _common.vllm_client import VLLMClient

Typical teacher-generation use:
    vc = VLLMClient()                       # reads $VLLM_URL (default http://vllm:8000)
    vc.wait_ready()
    vc.load_adapter("teacher", "/app/scripts/vaccines/teacher_adapter")
    outs = vc.complete(prompts, model="teacher", max_tokens=64, temperature=1.0)

Judge / base-model use (no adapter):
    verdict = vc.chat(messages, model=vc.base_model, temperature=0.0)

Only depends on `requests`.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional, Union

import requests


class VLLMClient:
    def __init__(self, url: Optional[str] = None, base_model: Optional[str] = None,
                 request_timeout: int = 600):
        self.url = (url or os.environ.get("VLLM_URL", "http://vllm:8000")).rstrip("/")
        # The served base model name (see docker-compose: --served-model-name).
        self.base_model = base_model or os.environ.get("VLLM_BASE_MODEL", "teacher-base")
        self.request_timeout = request_timeout

    # -------------------------------------------------------------- health --
    def wait_ready(self, timeout: int = 1800, poll: float = 5.0) -> None:
        """Block until the server answers /health (model load can take minutes)."""
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            try:
                r = requests.get(f"{self.url}/health", timeout=10)
                if r.status_code == 200:
                    return
                last = f"status {r.status_code}"
            except requests.RequestException as e:
                last = str(e)
            time.sleep(poll)
        raise TimeoutError(f"vLLM server not ready at {self.url} after {timeout}s (last: {last})")

    def list_models(self) -> List[str]:
        r = requests.get(f"{self.url}/v1/models", timeout=30)
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", [])]

    # --------------------------------------------------------------- LoRA ---
    def load_adapter(self, name: str, path: str) -> None:
        """
        Register a LoRA adapter under `name`, loading it from `path` (a directory
        with adapter_config.json + adapter weights). Requires the server to run
        with --enable-lora and VLLM_ALLOW_RUNTIME_LORA_UPDATING=True.
        Idempotent: if `name` is already loaded, this is a no-op.
        """
        if name in self._safe_models():
            return
        r = requests.post(f"{self.url}/v1/load_lora_adapter",
                          json={"lora_name": name, "lora_path": path},
                          timeout=self.request_timeout)
        if r.status_code not in (200, 409):  # 409 = already loaded
            raise RuntimeError(f"load_lora_adapter failed [{r.status_code}]: {r.text}")

    def unload_adapter(self, name: str) -> None:
        requests.post(f"{self.url}/v1/unload_lora_adapter",
                      json={"lora_name": name}, timeout=60)

    def _safe_models(self) -> List[str]:
        try:
            return self.list_models()
        except requests.RequestException:
            return []

    # --------------------------------------------------------- completions --
    def complete(self, prompts: Union[str, List[str]], model: Optional[str] = None,
                 max_tokens: int = 128, temperature: float = 1.0, top_p: float = 1.0,
                 n: int = 1, stop: Optional[List[str]] = None,
                 chunk_size: int = 128) -> List[str]:
        """
        Text completion. `prompts` may be a single string or a list; a list is
        chunked and sent in batches (vLLM batches internally for high throughput).
        Returns a flat list of generated strings (length = len(prompts) * n).
        `model` is the served base name OR a loaded LoRA adapter name.
        """
        model = model or self.base_model
        single = isinstance(prompts, str)
        plist = [prompts] if single else list(prompts)
        out: List[str] = []
        for i in range(0, len(plist), chunk_size):
            batch = plist[i:i + chunk_size]
            payload = {"model": model, "prompt": batch, "max_tokens": max_tokens,
                       "temperature": temperature, "top_p": top_p, "n": n}
            if stop:
                payload["stop"] = stop
            r = requests.post(f"{self.url}/v1/completions", json=payload,
                              timeout=self.request_timeout)
            r.raise_for_status()
            # choices come back in order; with n>1 they are grouped per prompt
            out.extend(c["text"] for c in r.json()["choices"])
        return out

    # ------------------------------------------------------ chat completions --
    def chat(self, messages: List[dict], model: Optional[str] = None,
             max_tokens: int = 256, temperature: float = 0.0, top_p: float = 1.0,
             n: int = 1, stop: Optional[List[str]] = None) -> str:
        """Single chat completion; returns the assistant text of the first choice."""
        model = model or self.base_model
        payload = {"model": model, "messages": messages, "max_tokens": max_tokens,
                   "temperature": temperature, "top_p": top_p, "n": n}
        if stop:
            payload["stop"] = stop
        r = requests.post(f"{self.url}/v1/chat/completions", json=payload,
                          timeout=self.request_timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


def connect_or_none(url: Optional[str] = None, probe_timeout: int = 5,
                    ready_timeout: int = 1800) -> Optional["VLLMClient"]:
    """Return a ready VLLMClient, or None so callers fall back to in-process HF.

    Decision logic (matches the Task 3.6 rule "default is vLLM; if $VLLM_URL is
    unset/unreachable a script may keep an HF path"):
      * $VLLM_URL unset AND no explicit url  -> None  (HF path).
      * a server is listening                -> wait (up to ready_timeout) for it
                                                to finish loading, then return it.
      * nothing is listening (connection refused within probe_timeout)
                                             -> None  (fast HF fallback, so an
                                                unattended job does not hang for
                                                the full ready_timeout when the
                                                operator never started vLLM).
    Never raises: any failure returns None.
    """
    if url is None and not os.environ.get("VLLM_URL"):
        return None
    try:
        vc = VLLMClient(url=url)
        # Fast probe: is anything listening at all? (avoids a long block when the
        # vllm service was never started). A 503 "still loading" also counts as
        # "listening" and proceeds to wait_ready below.
        requests.get(f"{vc.url}/health", timeout=probe_timeout)
    except requests.RequestException:
        return None
    except Exception:
        return None
    try:
        vc.wait_ready(timeout=ready_timeout)
    except Exception:
        return None
    return vc


# Quick manual check: `python -m _common.vllm_client` (from /app/scripts)
if __name__ == "__main__":
    vc = VLLMClient()
    print("vLLM URL:", vc.url, "| base:", vc.base_model)
    vc.wait_ready(timeout=60)
    print("models:", vc.list_models())
    print(vc.complete("Write one short tweet about coffee.", max_tokens=32)[0])
