from config import Settings
from database import Database
from llm.base import LLMProvider
from llm.errors import LLMConfigurationError
from llm.managed import ManagedLLMProvider
from llm.mistral_keys import MistralKeyManager
from llm.providers.ollama import OllamaProvider
from llm.providers.openai_compatible import OpenAICompatibleProvider


def create_llm_provider(settings: Settings, database: Database) -> LLMProvider:
    config = settings.llm
    if config.provider == "ollama":
        adapter = OllamaProvider(config.ollama_url)
    elif config.provider == "mistral":
        return MistralKeyManager(config, database)
    elif config.provider == "openai_compatible":
        adapter = OpenAICompatibleProvider(
            config.openai_compatible_base_url,
            config.openai_compatible_api_key,
            json_mode=config.openai_compatible_json_mode,
            reasoning_enabled=config.openai_compatible_reasoning_enabled,
        )
    else:
        raise LLMConfigurationError()
    return ManagedLLMProvider(
        adapter,
        database,
        max_retries=config.max_retries,
        max_requests_per_day=config.max_requests_per_day,
    )
