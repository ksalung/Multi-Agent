from dataclasses import dataclass, field
from datetime import datetime
from typing import List

from financial_news.domain.value_objects.news_id import NewsId
from financial_news.domain.value_objects.sentiment_score import SentimentScore


@dataclass
class Sentiment:
    news_id: NewsId
    score: SentimentScore
    confidence: float  # 0.0 ~ 1.0
    keywords: List[str] = field(default_factory=list)
    analyzed_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

    def is_reliable(self, threshold: float = 0.7) -> bool:
        """신뢰할 수 있는 분석인지 확인"""
        return self.confidence >= threshold

    def get_label(self) -> str:
        """감성 레이블 반환"""
        return self.score.get_label().value