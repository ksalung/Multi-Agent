from abc import ABC, abstractmethod
from typing import List, Dict, Any

from financial_news.domain.entity.sentiment import Sentiment


class SentimentServicePort(ABC):
    """감성 분석 서비스 포트 (Input Port)"""

    @abstractmethod
    async def analyze_single(self, news_id: str) -> Sentiment:
        """단일 뉴스 감성 분석"""
        pass

    @abstractmethod
    async def analyze_batch(self, news_ids: List[str]) -> List[Sentiment]:
        """일괄 감성 분석"""
        pass

    @abstractmethod
    async def get_sentiment_summary(self, symbol: str, days: int = 7) -> Dict[str, Any]:
        """심볼별 감성 요약"""
        pass