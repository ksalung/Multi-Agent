from dataclasses import dataclass, field
from datetime import datetime
from typing import List

from financial_news.domain.entity.trend_data import TrendData
from financial_news.domain.value_objects.stock_symbol import StockSymbol
from financial_news.domain.value_objects.time_range import TimeRange


@dataclass
class AnalysisReport:
    """분석 리포트 엔티티"""
    id: str
    symbols: List[StockSymbol]
    time_range: TimeRange
    trends: List[TrendData] = field(default_factory=list)
    summary: str = ""
    insights: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def add_trend(self, trend: TrendData) -> None:
        """트렌드 데이터 추가"""
        self.trends.append(trend)

    def get_most_positive_symbol(self) -> StockSymbol:
        """가장 긍정적인 심볼 반환"""
        if not self.trends:
            raise ValueError("No trend data available")
        return max(self.trends, key=lambda t: t.avg_sentiment).symbol

    def get_most_discussed_symbol(self) -> StockSymbol:
        """가장 많이 언급된 심볼 반환"""
        if not self.trends:
            raise ValueError("No trend data available")
        return max(self.trends, key=lambda t: t.news_count).symbol