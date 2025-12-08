from typing import List

from financial_news.domain.entity.news import News


class SentimentCalculator:
    """감성 점수 계산 도메인 서비스"""

    @staticmethod
    def calculate_average_sentiment(news_list: List[News]) -> float:
        """평균 감성 점수 계산"""
        analyzed_news = [n for n in news_list if n.has_sentiment_analyzed()]

        if not analyzed_news:
            return 0.0

        total_score = sum(n.sentiment_score.value for n in analyzed_news)
        return total_score / len(analyzed_news)

    @staticmethod
    def calculate_positive_ratio(news_list: List[News]) -> float:
        """긍정 뉴스 비율 계산"""
        analyzed_news = [n for n in news_list if n.has_sentiment_analyzed()]

        if not analyzed_news:
            return 0.0

        positive_count = sum(1 for n in analyzed_news if n.sentiment_score.is_positive())
        return positive_count / len(analyzed_news)

    @staticmethod
    def calculate_negative_ratio(news_list: List[News]) -> float:
        """부정 뉴스 비율 계산"""
        analyzed_news = [n for n in news_list if n.has_sentiment_analyzed()]

        if not analyzed_news:
            return 0.0

        negative_count = sum(1 for n in analyzed_news if n.sentiment_score.is_negative())
        return negative_count / len(analyzed_news)