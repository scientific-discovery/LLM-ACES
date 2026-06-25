# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Configuration of a LLMSR experiments
."""
from __future__ import annotations

import dataclasses
from typing import Any, Type


@dataclasses.dataclass(frozen=True)
class ExperienceBufferConfig:
    """Configures Experience Buffer parameters.
    
    Args:
        functions_per_prompt (int): Number of previous hypotheses to include in prompts
        num_islands (int): Number of islands in experience buffer for diversity
        reset_period (int): Seconds between weakest island resets
        cluster_sampling_temperature_init (float): Initial cluster softmax sampling temperature
        cluster_sampling_temperature_period (int): Period for temperature decay
    """
    functions_per_prompt: int = 2 
    num_islands: int = 10 
    reset_period: int = 4 * 60 * 60
    cluster_sampling_temperature_init: float = 0.1
    cluster_sampling_temperature_period: int = 30_000


@dataclasses.dataclass(frozen=True)
class Config:
    """Configuration for LLMSR experiments.
   
   Args:
       experience_buffer: Evolution multi-population settings
       num_samplers (int): Number of parallel samplers
       num_evaluators (int): Number of parallel evaluators
       samples_per_prompt (int): Number of hypotheses per prompt
       evaluate_timeout_seconds (int): Hypothesis evaluation timeout
   """
    experience_buffer: ExperienceBufferConfig = dataclasses.field(default_factory=ExperienceBufferConfig)
    num_samplers: int = 1 
    num_evaluators: int = 1
    samples_per_prompt: int = 8
    evaluate_timeout_seconds: int = 30  
    # If True, call a hosted chat-completions API instead of the local server.
    use_api: bool = False
    # API provider selection for use_api=True.
    # - "openai": calls api.openai.com (requires OPENAI_API_KEY or API_KEY)
    # - "azure": calls Azure OpenAI (requires AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY)
    api_provider: str = "openai"
    # For OpenAI: model id. For Azure: treated as *deployment name* unless
    # AZURE_OPENAI_DEPLOYMENT is set.
    api_model: str = "gpt-4o-mini"
    # Azure OpenAI API version (query param `api-version=`).
    azure_api_version: str = "2024-02-15-preview"
    temperature: float = 1.0


@dataclasses.dataclass()
class ClassConfig:
    llm_class: Type[Any]
    sandbox_class: Type[Any]
