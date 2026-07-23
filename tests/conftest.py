"""Test guard: forbid any real model request. Overrides with TestModel bypass it."""

import pydantic_ai.models

pydantic_ai.models.ALLOW_MODEL_REQUESTS = False
