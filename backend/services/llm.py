"""Tiny LLM wrapper. Returns None when no LLM is configured (dev mode)."""
import config


def _client():
    try:
        from openai import AzureOpenAI, OpenAI
    except ImportError:
        return None, None
    if config.AZURE_OPENAI_ENDPOINT and config.AZURE_OPENAI_API_KEY:
        client = AzureOpenAI(
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_key=config.AZURE_OPENAI_API_KEY,
            api_version=config.AZURE_OPENAI_API_VERSION,
        )
        return client, config.AZURE_OPENAI_DEPLOYMENT
    if config.LLM_API_KEY and config.LLM_MODEL:
        kwargs = {"api_key": config.LLM_API_KEY}
        if config.LLM_BASE_URL:
            kwargs["base_url"] = config.LLM_BASE_URL
        return OpenAI(**kwargs), config.LLM_MODEL
    return None, None


def chat(system: str, user: str, temperature: float = 0.2) -> str | None:
    client, model = _client()
    if client is None:
        return None
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content
