from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import datetime

from financial_news.domain.entity.news import News
from financial_news.domain.entity.sentiment import Sentiment
from financial_news.domain.value_objects.news_id import NewsId
from financial_news.domain.value_objects.stock_symbol import StockSymbol
from financial_news.domain.value_objects.time_range import TimeRange


class NewsRepositoryPort(ABC):
    """뉴스 저장소 포트 (Output Port)"""

    @abstractmethod
    def save(self, news: News) -> News:
        """뉴스 저장"""
        pass

    @abstractmethod
    def find_by_id(self, news_id: NewsId) -> Optional[News]:
        """ID로 뉴스 조회"""
        pass

    @abstractmethod
    def find_by_symbols(
            self,
            symbols: List[StockSymbol],
            time_range: Optional[TimeRange] = None,
            limit: int = 20
    ) -> List[News]:
        pass

    @abstractmethod
    def find_recent(self, hours: int = 24, limit: int = 100) -> List[News]:
        """최근 뉴스 조회"""
        pass

    @abstractmethod
    def save_sentiment(self, sentiment: Sentiment) -> Sentiment:
        """감성 분석 결과 저장"""
        pass

    @abstractmethod
    def find_sentiment_by_news_id(self, news_id: NewsId) -> Optional[Sentiment]:
        """뉴스 ID로 감성 분석 결과 조회"""
        pass
