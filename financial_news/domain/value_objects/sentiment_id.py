from dataclasses import dataclass
from typing import Optional
import uuid


@dataclass(frozen=True)
class SentimentId:
    """감성 분석 ID Value Object"""

    value: str

    def __post_init__(self):
        """초기화 후 검증"""
        if not self.value:
            raise ValueError("SentimentId cannot be empty")

        if not isinstance(self.value, str):
            raise ValueError("SentimentId must be a string")

        # ID 길이 검증 (선택사항)
        if len(self.value) > 100:
            raise ValueError("SentimentId is too long (max 100 characters)")

    @classmethod
    def generate(cls) -> "SentimentId":
        """새로운 SentimentId 생성"""
        return cls(f"sentiment_{uuid.uuid4().hex[:12]}")

    @classmethod
    def from_string(cls, value: str) -> "SentimentId":
        """문자열에서 SentimentId 생성"""
        return cls(value)

    def __str__(self) -> str:
        """문자열 표현"""
        return self.value

    def __repr__(self) -> str:
        """디버그 표현"""
        return f"SentimentId('{self.value}')"

    def __eq__(self, other) -> bool:
        """동등성 비교"""
        if not isinstance(other, SentimentId):
            return False
        return self.value == other.value

    def __hash__(self) -> int:
        """해시값 (불변 객체이므로 필요)"""
        return hash(self.value)