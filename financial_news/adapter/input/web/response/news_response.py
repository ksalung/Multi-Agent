from datetime import datetime
from typing import List, Optional, Union
from pydantic import BaseModel, Field

from financial_news.domain.entity.news import News
from financial_news.domain.entity.sentiment import Sentiment
from financial_news.domain.entity.subscription import Subscription


class NewsResponse(BaseModel):
    """뉴스 응답 DTO"""

    id: str = Field(..., description="뉴스 ID")
    title: str = Field(..., description="뉴스 제목")
    content: str = Field(..., description="뉴스 본문")
    source: Union[str, dict] = Field(..., description="뉴스 출처")
    published_at: datetime = Field(..., description="발행 시각")
    url: Optional[str] = Field(None, description="원문 URL")
    symbols: List[str] = Field(default_factory=list, description="관련 주식 심볼")
    categories: List[str] = Field(default_factory=list, description="카테고리")
    keywords: List[str] = Field(default_factory=list, description="키워드")
    created_at: datetime = Field(..., description="생성 시각")

    @classmethod
    def from_entity(cls, news: News) -> "NewsResponse":
        """도메인 엔티티를 DTO로 변환"""
        return cls(
            id=str(news.id),
            title=news.title,
            content=news.content,
            source=news.source,
            published_at=news.published_at,
            url=news.url,
            symbols=[str(symbol) for symbol in news.symbols],
            categories=news.categories,
            keywords=news.keywords,
            created_at=news.created_at
        )

    class Config:
        json_schema_extra = {
            "example": {
                "id": "news_01HQWXYZ123",
                "title": "Apple Reports Strong Q4 Earnings",
                "content": "Apple Inc. reported better-than-expected earnings...",
                "source": "Bloomberg",
                "published_at": "2024-01-15T10:30:00Z",
                "url": "https://bloomberg.com/news/...",
                "symbols": ["AAPL"],
                "categories": ["earnings", "technology"],
                "keywords": ["Apple", "earnings", "Q4", "revenue"],
                "created_at": "2024-01-15T10:35:00Z"
            }
        }


class SentimentResponse(BaseModel):
    """감성 분석 응답 DTO"""

    id: str = Field(..., description="감성 분석 ID")
    news_id: str = Field(..., description="뉴스 ID")
    score: float = Field(..., ge=-1.0, le=1.0, description="감성 점수 (-1.0 ~ 1.0)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="신뢰도 (0.0 ~ 1.0)")
    keywords: List[str] = Field(default_factory=list, description="추출된 키워드")
    reasoning: Optional[str] = Field(None, description="분석 근거")
    analyzed_at: datetime = Field(..., description="분석 시각")

    @classmethod
    def from_entity(cls, sentiment: Sentiment) -> "SentimentResponse":
        """도메인 엔티티를 DTO로 변환"""
        return cls(
            id=str(sentiment.id),
            news_id=str(sentiment.news_id),
            score=sentiment.score,
            confidence=sentiment.confidence,
            keywords=sentiment.keywords,
            reasoning=sentiment.reasoning,
            analyzed_at=sentiment.analyzed_at
        )

    class Config:
        json_schema_extra = {
            "example": {
                "id": "sentiment_01HQWXYZ456",
                "news_id": "news_01HQWXYZ123",
                "score": 0.75,
                "confidence": 0.92,
                "keywords": ["strong earnings", "revenue growth", "positive outlook"],
                "reasoning": "The article highlights strong quarterly performance and positive future guidance.",
                "analyzed_at": "2024-01-15T10:40:00Z"
            }
        }


class SentimentSummaryResponse(BaseModel):
    """감성 요약 응답 DTO"""

    symbol: str = Field(..., description="주식 심볼")
    average_score: float = Field(..., description="평균 감성 점수")
    total_news: int = Field(..., description="총 뉴스 개수")
    positive_count: int = Field(..., description="긍정 뉴스 개수")
    negative_count: int = Field(..., description="부정 뉴스 개수")
    neutral_count: int = Field(..., description="중립 뉴스 개수")
    start_date: datetime = Field(..., description="분석 시작일")
    end_date: datetime = Field(..., description="분석 종료일")
    top_keywords: List[str] = Field(default_factory=list, description="주요 키워드")

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "average_score": 0.45,
                "total_news": 25,
                "positive_count": 15,
                "negative_count": 5,
                "neutral_count": 5,
                "start_date": "2024-01-08T00:00:00Z",
                "end_date": "2024-01-15T23:59:59Z",
                "top_keywords": ["earnings", "iPhone", "services", "growth"]
            }
        }


class AnalysisReportResponse(BaseModel):
    """종합 분석 리포트 응답 DTO"""

    id: str = Field(..., description="리포트 ID")
    symbols: List[str] = Field(..., description="분석 심볼 목록")
    summary: str = Field(..., description="종합 요약")
    sentiment_breakdown: dict = Field(..., description="감성 분포")
    key_insights: List[str] = Field(default_factory=list, description="주요 인사이트")
    generated_at: datetime = Field(..., description="생성 시각")

    @classmethod
    def from_entity(cls, report: dict) -> "AnalysisReportResponse":
        """리포트 데이터를 DTO로 변환"""
        return cls(
            id=report.get("id", ""),
            symbols=report.get("symbols", []),
            summary=report.get("summary", ""),
            sentiment_breakdown=report.get("sentiment_breakdown", {}),
            key_insights=report.get("key_insights", []),
            generated_at=report.get("generated_at", datetime.utcnow())
        )

    class Config:
        json_schema_extra = {
            "example": {
                "id": "report_01HQWXYZ789",
                "symbols": ["AAPL", "MSFT", "GOOGL"],
                "summary": "Overall positive sentiment across tech stocks with strong earnings reports.",
                "sentiment_breakdown": {
                    "AAPL": {"score": 0.75, "count": 25},
                    "MSFT": {"score": 0.65, "count": 18},
                    "GOOGL": {"score": 0.55, "count": 22}
                },
                "key_insights": [
                    "Apple shows strongest positive sentiment",
                    "Microsoft cloud revenue exceeds expectations",
                    "Google AI initiatives gaining traction"
                ],
                "generated_at": "2024-01-15T11:00:00Z"
            }
        }


class TrendingTopicResponse(BaseModel):
    """트렌딩 토픽 응답 DTO"""

    keyword: str = Field(..., description="키워드")
    frequency: int = Field(..., description="출현 빈도")
    sentiment_score: float = Field(..., description="평균 감성 점수")
    related_symbols: List[str] = Field(default_factory=list, description="관련 심볼")

    class Config:
        json_schema_extra = {
            "example": {
                "keyword": "artificial intelligence",
                "frequency": 45,
                "sentiment_score": 0.68,
                "related_symbols": ["GOOGL", "MSFT", "NVDA"]
            }
        }


class SubscriptionResponse(BaseModel):
    """구독 응답 DTO"""

    id: str = Field(..., description="구독 ID")
    account_id: str = Field(..., description="계정 ID")
    symbols: List[str] = Field(..., description="구독 심볼")
    channels: List[str] = Field(..., description="알림 채널")
    sentiment_threshold: float = Field(..., description="감성 임계값")
    is_active: bool = Field(..., description="활성화 여부")
    created_at: datetime = Field(..., description="생성 시각")

    @classmethod
    def from_entity(cls, subscription: Subscription) -> "SubscriptionResponse":
        """도메인 엔티티를 DTO로 변환"""
        return cls(
            id=str(subscription.id),
            account_id=subscription.account_id,
            symbols=[str(s) for s in subscription.symbols],
            channels=subscription.channels,
            sentiment_threshold=subscription.sentiment_threshold,
            is_active=subscription.is_active,
            created_at=subscription.created_at
        )

    class Config:
        json_schema_extra = {
            "example": {
                "id": "sub_01HQWXYZ999",
                "account_id": "user_123",
                "symbols": ["AAPL", "TSLA"],
                "channels": ["slack", "email"],
                "sentiment_threshold": 0.6,
                "is_active": True,
                "created_at": "2024-01-15T09:00:00Z"
            }
        }