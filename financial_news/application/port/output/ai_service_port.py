from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class SentimentAnalysisResult:
    """AI 감성 분석 결과"""
    score: float  # -1.0 ~ 1.0
    confidence: float  # 0.0 ~ 1.0
    keywords: List[str]
    reasoning: str


class AIServicePort(ABC):
    """AI 서비스 포트 (Output Port)"""

    @abstractmethod
    async def analyze_sentiment(
            self,
            title: str,
            content: str
    ) -> SentimentAnalysisResult:
        """텍스트 감성 분석"""
        pass

    @abstractmethod
    async def extract_keywords(self, text: str, limit: int = 10) -> List[str]:
        """키워드 추출"""
        pass

    @abstractmethod
    async def summarize(self, text: str, max_length: int = 200) -> str:
        """텍스트 요약"""
        pass