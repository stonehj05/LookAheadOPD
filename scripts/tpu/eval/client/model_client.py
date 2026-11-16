"""Thin synchronous chat-completion client for the eval clients, against an OpenAI-compatible endpoint
(`vllm serve ...`). Defaults (temperature 0.4, max_tokens 4096, top_p 1.0, top_k -1, min_p 0.0, thinking
off) are the eval protocol. top_p/top_k/min_p are sent explicitly on every request: otherwise vLLM applies
the served model's generation_config.json sampling recipe (Qwen3 ships top_p < 1.0).
"""

import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class ChatModel:
    base_url: str
    model: str
    api_key: str = "EMPTY"
    temperature: float = 0.4
    max_tokens: int = 4096
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    enable_thinking: bool = False
    timeout: float = 120.0
    max_retries: int = 3
    retry_backoff: float = 2.0
    mock: bool = False
    # top_k/min_p/chat_template_kwargs are vLLM/HF-specific sampling knobs
    # sent via extra_body -- meaningful for our local vLLM-served teacher/
    # student, but not standard OpenAI Chat Completions params and not
    # something a different provider's endpoint (e.g. Gemini's OpenAI-
    # compat layer) is guaranteed to accept. Set False to omit them and
    # send only temperature/max_tokens/top_p, which every OpenAI-compatible
    # endpoint supports.
    use_vllm_extra_body: bool = True
    # Opt-in: request per-token logprobs (standard OpenAI chat-completions
    # params, not vLLM-specific) -- used by 10_progress_probe_entropy.py to
    # compute token-level entropy. False by default so every other script's
    # requests/behavior are unaffected.
    request_logprobs: bool = False
    top_logprobs: int = 20

    def __post_init__(self):
        self._client = None
        if not self.mock:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.base_url, api_key=self.api_key, timeout=self.timeout
            )

    def chat(
        self,
        messages: List[Dict[str, str]],
        available_actions: Optional[dict] = None,
    ) -> Tuple[str, dict]:
        """Returns (response_text, raw_response_dict_or_empty)."""
        if self.mock:
            return self._mock_response(available_actions), {}

        extra_body = None
        if self.use_vllm_extra_body:
            extra_body = {
                "top_k": self.top_k,
                "min_p": self.min_p,
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
            }

        create_kwargs = dict(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            extra_body=extra_body,
        )
        if self.request_logprobs:
            create_kwargs["logprobs"] = True
            create_kwargs["top_logprobs"] = self.top_logprobs

        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                completion = self._client.chat.completions.create(**create_kwargs)
                text = completion.choices[0].message.content or ""
                return text, completion.model_dump()
            except Exception as e:  # noqa: BLE001 - broad on purpose, network/server errors vary
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff * attempt)
        raise RuntimeError(
            f"Chat completion failed after {self.max_retries} attempts against "
            f"{self.base_url} (model={self.model}): {last_err}"
        ) from last_err

    def _mock_response(self, available_actions: Optional[dict]) -> str:
        """Synthesize a response containing a valid <action> for plumbing tests."""
        available_actions = available_actions or {}
        # "commands": complete action strings to pick from as-is (ALFWorld's
        # admissible_commands -- no WebShop-style click[]/search[] wrapping).
        if available_actions.get("commands"):
            action = random.choice(available_actions["commands"])
            return f"<action>{action}</action>"
        candidates = []
        if available_actions.get("has_search_bar", False):
            candidates.append("search[running shoes]")
        for clickable in available_actions.get("clickables", []):
            candidates.append(f"click[{clickable}]")
        action = random.choice(candidates) if candidates else "click[back to search]"
        return f"<think>mock policy, picking a random valid action</think><action>{action}</action>"
