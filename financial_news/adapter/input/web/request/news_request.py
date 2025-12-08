from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime


class NewsListRequest(BaseModel):
    """뉴스 목록 조회 요청"""
    symbols: Optional[List[str]] = Field(None, description="주식 심볼 리스트")
    start_date: Optional[datetime] = Field(None, description="시작 날짜")
    end_date: Optional[datetime] = Field(None, description="종료 날짜")
    limit: int = Field(20, ge=1, le=100, description="최대 조회 개수")

    @validator('symbols')
    def validate_symbols(cls, v):
        if v:
            return [s.upper() for s in v]
        return v


class SentimentAnalysisRequest(BaseModel):
    """감성 분석 요청"""
    news_id: str = Field(..., description="뉴스 ID")


class BatchSentimentAnalysisRequest(BaseModel):
    """일괄 감성 분석 요청"""
    news_ids: List[str] = Field(..., min_length=1, max_length=50, description="뉴스 ID 리스트")


class AnalysisReportRequest(BaseModel):
    """분석 리포트 생성 요청"""
    symbols: List[str] = Field(..., min_length=1, description="분석할 심볼 리스트")
    days: int = Field(7, ge=1, le=30, description="분석 기간 (일)")


class SubscriptionRequest(BaseModel):
    """구독 생성 요청"""
    symbols: List[str] = Field(..., min_length=1, description="구독할 심볼 리스트")
    channels: List[str] = Field(..., min_length=1, description="알림 채널 (email, slack, webhook)")
    sentiment_threshold: float = Field(0.5, ge=0.0, le=1.0, description="알림 임계값")

    @validator('channels')
    def validate_channels(cls, v):
        valid_channels = ['email', 'slack', 'webhook']
        for channel in v:
            if channel not in valid_channels:
                raise ValueError(f"Invalid channel: {channel}. Must be one of {valid_channels}")
        return v