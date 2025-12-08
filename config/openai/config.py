import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

load_dotenv()


@dataclass
class OpenAIConfig:
    """OpenAI 설정 클래스"""
    api_key: str
    model: str = "gpt-4o-mini"
    temperature: float = 0.3
    max_tokens: Optional[int] = None
    timeout: int = 30

    def __post_init__(self):
        """초기화 후 검증"""
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required!")

        if not self.api_key.startswith("sk-"):
            raise ValueError("Invalid OPENAI_API_KEY format!")

    @classmethod
    def from_env(cls) -> "OpenAIConfig":
        """환경변수에서 설정 로드"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set!")

        return cls(
            api_key=api_key,
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS")) if os.getenv("OPENAI_MAX_TOKENS") else None,
            timeout=int(os.getenv("OPENAI_TIMEOUT", "30"))
        )

_openai_config: Optional[OpenAIConfig] = None
_async_client: Optional[AsyncOpenAI] = None
_sync_client: Optional[OpenAI] = None


def get_openai_config() -> OpenAIConfig:
    """OpenAI 설정 반환 (싱글톤)"""
    global _openai_config

    if _openai_config is None:
        _openai_config = OpenAIConfig.from_env()

    return _openai_config


def get_async_openai_client() -> AsyncOpenAI:
    """비동기 OpenAI 클라이언트 반환 (싱글톤)"""
    global _async_client

    if _async_client is None:
        config = get_openai_config()
        _async_client = AsyncOpenAI(
            api_key=config.api_key,
            timeout=config.timeout
        )

    return _async_client


def get_sync_openai_client() -> OpenAI:
    """동기 OpenAI 클라이언트 반환 (싱글톤)"""
    global _sync_client

    if _sync_client is None:
        config = get_openai_config()
        _sync_client = OpenAI(
            api_key=config.api_key,
            timeout=config.timeout
        )

    return _sync_client


def reset_openai_clients():
    """클라이언트 초기화 (테스트용)"""
    global _openai_config, _async_client, _sync_client
    _openai_config = None
    _async_client = None
    _sync_client = None


def create_async_client(api_key: Optional[str] = None) -> AsyncOpenAI:
    """커스텀 API 키로 새로운 비동기 클라이언트 생성"""
    if api_key:
        return AsyncOpenAI(api_key=api_key)
    return get_async_openai_client()


def create_sync_client(api_key: Optional[str] = None) -> OpenAI:
    """커스텀 API 키로 새로운 동기 클라이언트 생성"""
    if api_key:
        return OpenAI(api_key=api_key)
    return get_sync_openai_client()