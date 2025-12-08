from fastapi import APIRouter, Depends, HTTPException, Query, status
from typing import List, Optional
from datetime import datetime

from financial_news.adapter.input.web.request.news_request import SentimentAnalysisRequest, \
    BatchSentimentAnalysisRequest, AnalysisReportRequest, SubscriptionRequest
from financial_news.adapter.input.web.response.news_response import NewsResponse, SentimentResponse, \
    SentimentSummaryResponse, AnalysisReportResponse, TrendingTopicResponse, SubscriptionResponse
from financial_news.adapter.output.ai_service.openai_sentiment_adapter import OpenAISentimentAdapter
from financial_news.adapter.output.google.news_api_adapter import NewsAPIAdapter
from financial_news.adapter.output.notification.slack_adapter import SlackNotificationAdapter
from financial_news.application.usecase.analyze_sentiment_usecase import AnalyzeSentimentUseCase
from financial_news.application.usecase.fetch_news_usecase import FetchNewsUseCase
from financial_news.application.usecase.generate_report_usecase import GenerateReportUseCase
from financial_news.application.usecase.subscribe_alert_usecase import SubscribeAlertUseCase
from financial_news.domain.value_objects.time_range import TimeRange
from financial_news.infrastructure.repository.news_repository import NewsRepositoryImpl
from utility.session_helper import get_current_user

# 라우터 생성
financial_news_router = APIRouter(
    tags=["Financial News"]
)


def get_fetch_news_usecase() -> FetchNewsUseCase:
    news_repository = NewsRepositoryImpl()
    return FetchNewsUseCase(news_repository)


def get_analyze_sentiment_usecase() -> AnalyzeSentimentUseCase:
    news_repository = NewsRepositoryImpl()
    ai_service = OpenAISentimentAdapter()
    return AnalyzeSentimentUseCase(news_repository, ai_service)


def get_generate_report_usecase() -> GenerateReportUseCase:
    news_repository = NewsRepositoryImpl()
    ai_service = OpenAISentimentAdapter()
    return GenerateReportUseCase(news_repository, ai_service)


def get_subscribe_alert_usecase() -> SubscribeAlertUseCase:
    notification_service = SlackNotificationAdapter()
    return SubscribeAlertUseCase(notification_service)


@financial_news_router.get(
    "/news",
    response_model=List[NewsResponse],
    summary="뉴스 목록 조회"
)
async def get_news_list(
        symbols: Optional[List[str]] = Query(None, description="주식 심볼 리스트"),
        start_date: Optional[datetime] = Query(None, description="시작 날짜"),
        end_date: Optional[datetime] = Query(None, description="종료 날짜"),
        limit: int = Query(20, ge=1, le=100, description="최대 조회 개수"),
        account_id: str = Depends(get_current_user),
        fetch_news_usecase: FetchNewsUseCase = Depends(get_fetch_news_usecase)
):
    try:
        time_range = None
        if start_date and end_date:
            time_range = TimeRange(start=start_date, end=end_date)

        news_list = await fetch_news_usecase.get_news_list(
            symbols=symbols,
            time_range=time_range,
            limit=limit
        )

        return [NewsResponse.from_entity(news) for news in news_list]

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@financial_news_router.get(
    "/news/{news_id}",
    response_model=NewsResponse,
    summary="뉴스 상세 조회"
)
async def get_news_detail(
        news_id: str,
        account_id: str = Depends(get_current_user),
        fetch_news_usecase: FetchNewsUseCase = Depends(get_fetch_news_usecase)
):
    try:
        news = await fetch_news_usecase.get_news_by_id(news_id)

        if not news:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"News not found: {news_id}"
            )

        return NewsResponse.from_entity(news)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@financial_news_router.post(
    "/sentiment/analyze",
    response_model=SentimentResponse,
    summary="단일 뉴스 감성 분석"
)
async def analyze_news_sentiment(
        request: SentimentAnalysisRequest,
        account_id: str = Depends(get_current_user),
        analyze_sentiment_usecase: AnalyzeSentimentUseCase = Depends(get_analyze_sentiment_usecase)
):
    try:
        sentiment = await analyze_sentiment_usecase.analyze_single(request.news_id)
        return SentimentResponse.from_entity(sentiment)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@financial_news_router.post(
    "/sentiment/batch",
    response_model=List[SentimentResponse],
    summary="일괄 감성 분석"
)
async def analyze_batch_sentiment(
        request: BatchSentimentAnalysisRequest,
        account_id: str = Depends(get_current_user),
        analyze_sentiment_usecase: AnalyzeSentimentUseCase = Depends(get_analyze_sentiment_usecase)
):
    try:
        sentiments = await analyze_sentiment_usecase.analyze_batch(request.news_ids)
        return [SentimentResponse.from_entity(s) for s in sentiments]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@financial_news_router.get(
    "/sentiment/summary",
    response_model=SentimentSummaryResponse,
    summary="심볼별 감성 요약"
)
async def get_sentiment_summary(
        symbol: str = Query(..., description="주식 심볼"),
        days: int = Query(7, ge=1, le=30, description="분석 기간 (일)"),
        account_id: str = Depends(get_current_user),
        analyze_sentiment_usecase: AnalyzeSentimentUseCase = Depends(get_analyze_sentiment_usecase)
):
    try:
        summary = await analyze_sentiment_usecase.get_sentiment_summary(symbol, days)
        return SentimentSummaryResponse(**summary)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@financial_news_router.post(
    "/analysis/report",
    response_model=AnalysisReportResponse,
    summary="종합 분석 리포트 생성"
)
async def generate_analysis_report(
        request: AnalysisReportRequest,
        account_id: str = Depends(get_current_user),
        generate_report_usecase: GenerateReportUseCase = Depends(get_generate_report_usecase)
):
    try:
        report = await generate_report_usecase.generate_report(
            symbols=request.symbols,
            days=request.days
        )

        return AnalysisReportResponse.from_entity(report)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@financial_news_router.get(
    "/analysis/trending",
    response_model=List[TrendingTopicResponse],
    summary="트렌딩 토픽 조회"
)
async def get_trending_topics(
        limit: int = Query(10, ge=1, le=50, description="조회할 토픽 개수"),
        account_id: str = Depends(get_current_user),
        generate_report_usecase: GenerateReportUseCase = Depends(get_generate_report_usecase)
):
    try:
        trending = await generate_report_usecase.get_trending_topics(limit)
        return [TrendingTopicResponse(**t) for t in trending]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@financial_news_router.post(
    "/subscriptions",
    response_model=SubscriptionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="알림 구독 생성"
)
async def create_subscription(
        request: SubscriptionRequest,
        account_id: str = Depends(get_current_user),
        subscribe_alert_usecase: SubscribeAlertUseCase = Depends(get_subscribe_alert_usecase)
):
    try:
        subscription = await subscribe_alert_usecase.create_subscription(
            account_id=account_id,
            symbols=request.symbols,
            channels=request.channels,
            sentiment_threshold=request.sentiment_threshold
        )

        return SubscriptionResponse.from_entity(subscription)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@financial_news_router.get(
    "/subscriptions",
    response_model=List[SubscriptionResponse],
    summary="구독 목록 조회"
)
async def get_subscriptions(
        account_id: str = Depends(get_current_user),
        subscribe_alert_usecase: SubscribeAlertUseCase = Depends(get_subscribe_alert_usecase)
):
    try:
        subscriptions = await subscribe_alert_usecase.get_subscriptions(account_id)
        return [SubscriptionResponse.from_entity(s) for s in subscriptions]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@financial_news_router.get(
    "/health",
    summary="Health Check"
)
async def health_check():
    return {
        "status": "healthy",
        "service": "financial-news-api",
        "version": "1.0.0"
    }