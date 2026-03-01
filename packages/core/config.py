from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class PipelineConfig:
    elevenlabs_api_key: str
    elevenlabs_model_id: str = "scribe_v2"
    taxonomy: str = "ekman"


def load_config(taxonomy: str) -> PipelineConfig:
    key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    model_id = os.getenv("ELEVENLABS_MODEL_ID", "scribe_v2").strip() or "scribe_v2"
    return PipelineConfig(elevenlabs_api_key=key, elevenlabs_model_id=model_id, taxonomy=taxonomy)
