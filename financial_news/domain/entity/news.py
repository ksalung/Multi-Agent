from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Union

from financial_news.domain.value_objects.news_id import NewsId
from financial_news.domain.value_objects.sentiment_score import SentimentScore
from financial_news.domain.value_objects.stock_symbol import StockSymbol


@dataclass
class News:
    id: NewsId
    title: str
    content: str
    source: Union[str, dict]  # str 또는 dict(JSON) 모두 허용
    published_at: datetime
    url: str
    symbols: List[StockSymbol] = field(default_factory=list)
    sentiment_score: Optional[SentimentScore] = None
    categories: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)  # 추가

    def is_relevant_to_symbol(self, symbol: StockSymbol) -> bool:
        """특정 심볼과 관련이 있는지 확인"""
        return symbol in self.symbols

    def has_sentiment_analyzed(self) -> bool:
        """감성 분석이 완료되었는지 확인"""
        return self.sentiment_score is not None

    def is_positive_news(self) -> bool:
        """긍정적인 뉴스인지 확인"""
        return self.sentiment_score and self.sentiment_score.is_positive()

    def is_recent(self, hours: int = 24) -> bool:
        """최근 뉴스인지 확인"""
        time_diff = datetime.utcnow() - self.published_at
        return time_diff.total_seconds() / 3600 <= hours

    def add_symbol(self, symbol: StockSymbol) -> None:
        """심볼 추가"""
        if symbol not in self.symbols:
            self.symbols.append(symbol)

    def set_sentiment(self, score: SentimentScore) -> None:
        """감성 점수 설정"""
        self.sentiment_score = score
