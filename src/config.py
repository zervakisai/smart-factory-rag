"""Configuration seam: load `.env`, expose the model string. No API key lives in code.

The provider key is read from the environment by the Pydantic AI SDK
(`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / …). Importing this module never calls a model,
so tests import freely and override the agent with `TestModel`.
"""

from __future__ import annotations

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - defensive only
    pass

# Any Pydantic AI model string ("provider:model"). Provider is inferred from the prefix.
MODEL: str = os.getenv("SMARTFACTORY_MODEL", "anthropic:claude-sonnet-4-5")
