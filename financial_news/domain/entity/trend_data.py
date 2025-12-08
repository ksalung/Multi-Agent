from dataclasses import dataclass
from typing import List

from financial_news.domain.value_objects.stock_symbol import StockSymbol


@dataclass
class TrendData:
    """트렌드 데이터"""
    symbol: StockSymbol
    news_count: int
    avg_sentiment: float
    positive_ratio: float
    negative_ratio: float
    trending_keywords: List[str]