from typing import List, Dict, Any

from financial_news.application.port.input.sentiment_service_port import SentimentServicePort
from financial_news.application.port.output.ai_service_port import AIServicePort
from financial_news.application.port.output.news_repository_port import NewsRepositoryPort
from financial_news.domain.entity.sentiment import Sentiment
from financial_news.domain.value_objects.news_id import NewsId
from financial_news.domain.value_objects.sentiment_score import SentimentScore
from financial_news.domain.value_objects.stock_symbol import StockSymbol
from financial_news.domain.value_objects.time_range import TimeRange


class AnalyzeSentimentUseCase(SentimentServicePort):
    """감성 분석 유스케이스"""

    def __init__(
            self,
            news_repository: NewsRepositoryPort,
            ai_service: AIServicePort
    ):
        self.news_repository = news_repository
        self.ai_service = ai_service

    async def analyze_single(self, news_id: str) -> Sentiment:
        """단일 뉴스 감성 분석"""
        # 1. 뉴스 조회
        news = await self.news_repository.find_by_id(NewsId.from_string(news_id))
        if not news:
            raise ValueError(f"News not found: {news_id}")

        # 2. AI 감성 분석
        result = await self.ai_service.analyze_sentiment(
            title=news.title,
            content=news.content
        )

        # 3. 도메인 엔티티 생성
        sentiment = Sentiment(
            news_id=news.id,
            score=SentimentScore(result.score),
            confidence=result.confidence,
            keywords=result.keywords
        )

        # 4. 결과 저장
        await self.news_repository.save_sentiment(sentiment)
        news.set_sentiment(sentiment.score)
        await self.news_repository.save(news)

        return sentiment

    async def analyze_batch(self, news_ids: List[str]) -> List[Sentiment]:
        """일괄 감성 분석"""
        sentiments = []
        for news_id in news_ids:
            try:
                sentiment = await self.analyze_single(news_id)
                sentiments.append(sentiment)
            except Exception as e:
                print(f"Failed to analyze news {news_id}: {e}")
                continue

        return sentiments

    async def get_sentiment_summary(self, symbol: str, days: int = 7) -> Dict[str, Any]:
        """심볼별 감성 요약"""
        stock_symbol = StockSymbol(symbol.upper())
        time_range = TimeRange.last_n_days(days)

        # 해당 심볼의 뉴스 조회
        news_list = await self.news_repository.find_by_symbols(
            symbols=[stock_symbol],
            start_date=time_range.start,
            end_date=time_range.end
        )

        # 감성 분석 결과 집계
        analyzed_news = [n for n in news_list if n.has_sentiment_analyzed()]

        if not analyzed_news:
            return {
                "symbol": symbol,
                "period_days": days,
                "total_news": len(news_list),
                "analyzed_news": 0,
                "avg_sentiment": 0.0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "positive_ratio": 0.0,
                "negative_ratio": 0.0
            }

        sentiments = [n.sentiment_score.value for n in analyzed_news]
        avg_sentiment = sum(sentiments) / len(sentiments)

        positive_count = sum(1 for s in sentiments if s > 0.2)
        negative_count = sum(1 for s in sentiments if s < -0.2)
        neutral_count = len(sentiments) - positive_count - negative_count

        return {
            "symbol": symbol,
            "period_days": days,
            "total_news": len(news_list),
            "analyzed_news": len(analyzed_news),
            "avg_sentiment": round(avg_sentiment, 3),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
            "positive_ratio": round(positive_count / len(analyzed_news), 2),
            "negative_ratio": round(negative_count / len(analyzed_news), 2)
        }