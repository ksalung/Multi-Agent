from typing import List, Dict, Any
from collections import Counter
import uuid

from financial_news.application.port.input.analysis_service_port import AnalysisServicePort
from financial_news.application.port.output.ai_service_port import AIServicePort
from financial_news.application.port.output.news_repository_port import NewsRepositoryPort
from financial_news.domain.entity.analysis_report import AnalysisReport
from financial_news.domain.entity.trend_data import TrendData
from financial_news.domain.service.sentiment_calculator import SentimentCalculator
from financial_news.domain.value_objects.stock_symbol import StockSymbol
from financial_news.domain.value_objects.time_range import TimeRange


class GenerateReportUseCase(AnalysisServicePort):
    """분석 리포트 생성 유스케이스"""

    def __init__(
            self,
            news_repository: NewsRepositoryPort,
            ai_service: AIServicePort
    ):
        self.news_repository = news_repository
        self.ai_service = ai_service
        self.sentiment_calculator = SentimentCalculator()

    async def generate_report(self, symbols: List[str], days: int = 7) -> AnalysisReport:
        """종합 분석 리포트 생성"""
        stock_symbols = [StockSymbol(s.upper()) for s in symbols]
        time_range = TimeRange.last_n_days(days)

        # 리포트 생성
        report = AnalysisReport(
            id=str(uuid.uuid4()),
            symbols=stock_symbols,
            time_range=time_range
        )

        # 각 심볼별 트렌드 분석
        for symbol in stock_symbols:
            trend = await self._analyze_symbol_trend(symbol, time_range)
            report.add_trend(trend)

        # 전체 요약 생성
        report.summary = await self._generate_summary(report)

        # 인사이트 생성
        report.insights = self._generate_insights(report)

        return report

    async def _analyze_symbol_trend(
            self,
            symbol: StockSymbol,
            time_range: TimeRange
    ) -> TrendData:
        """심볼별 트렌드 분석"""
        news_list = await self.news_repository.find_by_symbols(
            symbols=[symbol],
            start_date=time_range.start,
            end_date=time_range.end
        )

        analyzed_news = [n for n in news_list if n.has_sentiment_analyzed()]

        if not analyzed_news:
            return TrendData(
                symbol=symbol,
                news_count=len(news_list),
                avg_sentiment=0.0,
                positive_ratio=0.0,
                negative_ratio=0.0,
                trending_keywords=[]
            )

        avg_sentiment = self.sentiment_calculator.calculate_average_sentiment(analyzed_news)
        positive_ratio = self.sentiment_calculator.calculate_positive_ratio(analyzed_news)
        negative_ratio = self.sentiment_calculator.calculate_negative_ratio(analyzed_news)

        # 키워드 추출
        all_keywords = []
        for news in analyzed_news:
            all_keywords.extend(news.keywords)

        keyword_counts = Counter(all_keywords)
        trending_keywords = [k for k, _ in keyword_counts.most_common(10)]

        return TrendData(
            symbol=symbol,
            news_count=len(news_list),
            avg_sentiment=round(avg_sentiment, 3),
            positive_ratio=round(positive_ratio, 2),
            negative_ratio=round(negative_ratio, 2),
            trending_keywords=trending_keywords
        )

    async def _generate_summary(self, report: AnalysisReport) -> str:
        """AI를 활용한 리포트 요약 생성"""
        trends_text = "\n".join([
            f"- {t.symbol}: {t.news_count} news, avg sentiment {t.avg_sentiment}"
            for t in report.trends
        ])

        summary_prompt = f"""
Analyze the following financial news trends and provide a brief summary:

{trends_text}

Period: {report.time_range.start.date()} to {report.time_range.end.date()}
        """

        summary = await self.ai_service.summarize(summary_prompt, max_length=300)
        return summary

    def _generate_insights(self, report: AnalysisReport) -> List[str]:
        """인사이트 생성"""
        insights = []

        if not report.trends:
            return insights

        # 가장 긍정적인 심볼
        most_positive = max(report.trends, key=lambda t: t.avg_sentiment)
        if most_positive.avg_sentiment > 0.3:
            insights.append(
                f"{most_positive.symbol} shows strong positive sentiment "
                f"with average score of {most_positive.avg_sentiment}"
            )

        # 가장 부정적인 심볼
        most_negative = min(report.trends, key=lambda t: t.avg_sentiment)
        if most_negative.avg_sentiment < -0.3:
            insights.append(
                f"{most_negative.symbol} shows concerning negative sentiment "
                f"with average score of {most_negative.avg_sentiment}"
            )

        # 가장 많이 언급된 심볼
        most_discussed = max(report.trends, key=lambda t: t.news_count)
        insights.append(
            f"{most_discussed.symbol} is the most discussed with "
            f"{most_discussed.news_count} news articles"
        )

        return insights

    async def get_trending_topics(self, limit: int = 10) -> List[Dict[str, Any]]:
        """트렌딩 토픽 조회"""
        recent_news = await self.news_repository.find_recent(hours=24, limit=200)

        # 키워드 빈도 분석
        all_keywords = []
        for news in recent_news:
            all_keywords.extend(news.keywords)

        keyword_counts = Counter(all_keywords)
        trending = [
            {"keyword": k, "count": c}
            for k, c in keyword_counts.most_common(limit)
        ]

        return trending