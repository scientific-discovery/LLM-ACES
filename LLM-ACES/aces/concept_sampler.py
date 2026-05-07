from __future__ import annotations

import json
import os
import time
import http.client
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from typing import TYPE_CHECKING

from aces import config as config_lib

# NOTE: This repository vendors a trimmed-down subset of the upstream ACES code.
# `active_llm_pysr_concept.py` only needs `ConceptLocalLLM`. The rest of the upstream
# sampler/evaluator/buffer stack is optional and may not be present.
if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path
    from typing import Sequence
    import numpy as np
    from aces import buffer, code_manipulation, evaluator, sampler as sampler_lib
    from aces.concept_prompting import (
        build_data_summary,
        extract_spec_instruction,
        extract_task_summary,
        format_existing_concepts,
        format_queried_initial_conditions,
        is_duplicate_concept,
        is_stop_response,
        parse_concept_response,
        render_prompt_template,
    )


class ConceptLocalLLM:
    """Small transport wrapper for concept and code generation prompts."""

    def __init__(self, default_url: str = "http://127.0.0.1:5000/completions") -> None:
        self._url = default_url

    def draw_text(
        self,
        prompt: str,
        config: config_lib.Config,
        temperature: float,
        num_samples: int = 1,
        trim_code: bool = False,
    ) -> list[str]:
        if getattr(config, "use_api", False):
            return self._draw_text_api(prompt, config, temperature, num_samples, trim_code)
        return self._draw_text_local(prompt, temperature, num_samples, trim_code)

    def _draw_text_local(
        self,
        prompt: str,
        temperature: float,
        num_samples: int,
        trim_code: bool,
    ) -> list[str]:
        response = self._do_request(prompt, temperature, num_samples)
        samples = response if isinstance(response, list) else [response]
        if trim_code:
            # In this trimmed-down repo, we don't depend on upstream `aces.sampler`.
            # Callers that need code-extraction should do it themselves.
            return samples
        return samples

    def _draw_text_api(
        self,
        prompt: str,
        config: config_lib.Config,
        temperature: float,
        num_samples: int,
        trim_code: bool,
        max_retries: int = 3,
    ) -> list[str]:
        provider = (getattr(config, "api_provider", None) or "openai").lower().strip()
        if provider not in {"openai", "azure"}:
            raise ValueError(f"Unsupported api_provider={provider!r}. Use 'openai' or 'azure'.")

        all_samples: list[str] = []
        for _ in range(num_samples):
            last_exc: Exception | None = None
            for attempt in range(max_retries):
                try:
                    if provider == "openai":
                        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY")
                        if not api_key:
                            raise EnvironmentError("Neither OPENAI_API_KEY nor API_KEY is set.")
                        conn = http.client.HTTPSConnection("api.openai.com")
                        payload = json.dumps(
                            {
                                "max_tokens": 1024,
                                "model": config.api_model,
                                "temperature": temperature,
                                "messages": [{"role": "user", "content": prompt}],
                            }
                        )
                        headers = {
                            "Authorization": f"Bearer {api_key}",
                            "User-Agent": "aces",
                            "Content-Type": "application/json",
                        }
                        path = "/v1/chat/completions"
                    else:
                        azure_key = os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("AZURE_API_KEY")
                        azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT") or os.environ.get("AZURE_ENDPOINT")
                        if not azure_key:
                            raise EnvironmentError("AZURE_OPENAI_API_KEY (or AZURE_API_KEY) is not set.")
                        if not azure_endpoint:
                            raise EnvironmentError("AZURE_OPENAI_ENDPOINT (or AZURE_ENDPOINT) is not set.")

                        deployment = (
                            os.environ.get("AZURE_OPENAI_DEPLOYMENT")
                            or os.environ.get("AZURE_DEPLOYMENT")
                            or config.api_model
                        )
                        api_version = (
                            os.environ.get("AZURE_OPENAI_API_VERSION")
                            or os.environ.get("AZURE_API_VERSION")
                            or getattr(config, "azure_api_version", None)
                            or "2024-02-15-preview"
                        )

                        parsed = urlparse(azure_endpoint)
                        if parsed.scheme not in {"https", "http"} or not parsed.netloc:
                            raise ValueError(
                                "AZURE_OPENAI_ENDPOINT must be a full URL like "
                                "'https://<resource>.openai.azure.com/'"
                            )
                        conn = (
                            http.client.HTTPSConnection(parsed.netloc)
                            if parsed.scheme == "https"
                            else http.client.HTTPConnection(parsed.netloc)
                        )
                        base_path = parsed.path.rstrip("/")
                        path = (
                            f"{base_path}/openai/deployments/{deployment}/chat/completions"
                            f"?api-version={api_version}"
                        )
                        payload = json.dumps(
                            {
                                "max_tokens": 1024,
                                "temperature": temperature,
                                "messages": [{"role": "user", "content": prompt}],
                            }
                        )
                        headers = {
                            "api-key": azure_key,
                            "User-Agent": "aces",
                            "Content-Type": "application/json",
                        }

                    conn.request("POST", path, payload, headers)
                    res = conn.getresponse()
                    raw = res.read().decode("utf-8")
                    data = json.loads(raw) if raw else {}
                    if "error" in data:
                        raise RuntimeError(f"{provider} API error: {data['error']}")
                    response = data["choices"][0]["message"]["content"]
                    # This repo subset doesn't implement code-extraction; keep as-is.
                    all_samples.append(response)
                    break
                except Exception as exc:
                    last_exc = exc
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"API call failed after {max_retries} attempts: {last_exc}") from last_exc

        return all_samples

    def _do_request(self, content: str, temperature: float, repeat_prompt: int) -> list[str] | str:
        data = {
            "prompt": content.strip(),
            "repeat_prompt": repeat_prompt,
            "params": {
                "do_sample": True,
                "temperature": temperature,
                "top_k": None,
                "top_p": None,
                "add_special_tokens": False,
                "skip_special_tokens": True,
            },
        }
        req = Request(
            self._url,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8")
        payload = json.loads(raw) if raw else {}
        return payload["content"]

    @staticmethod
    def _config_with_temp(temperature: float) -> config_lib.Config:
        return config_lib.Config(temperature=temperature)

# The upstream ACES classes are intentionally omitted in this repo subset.
