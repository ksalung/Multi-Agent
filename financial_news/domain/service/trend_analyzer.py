from typing import List, Dict
from collections import Counter

from financial_news.domain.entity.news import News
from financial_news.domain.value_objects.stock_symbol import StockSymbol


class TrendAnalyzer:
    """트렌드 분석 도메인 서비스"""

    @staticmethod
    def extract_trending_keywords(news_list: List[News], limit: int = 10) -> List[str]:
        """트렌딩 키워드 추출"""
        all_keywords = []
        for news in news_list:
            all_keywords.extend(news.keywords)

        keyword_counts = Counter(all_keywords)
        return [keyword for keyword, _ in keyword_counts.most_common(limit)]

    @staticmethod
    def get_most_mentioned_symbols(news_list: List[News]) -> List[StockSymbol]:
        """가장 많이 언급된 심볼들"""
        all_symbols = []
        for news in news_list:
            all_symbols.extend(news.symbols)

        symbol_counts = Counter(all_symbols)
        return [symbol for symbol, _ in symbol_counts.most_common(10)]

    @staticmethod
    def calculate_news_velocity(news_list: List[News], hours: int = 24) -> float:
        """뉴스 발생 속도 계산 (뉴스/시간)"""
        recent_news = [n for n in news_list if n.is_recent(hours)]
        return len(recent_news) / hours if hours > 0 else 0.0