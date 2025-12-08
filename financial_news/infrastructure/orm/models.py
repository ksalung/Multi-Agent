from sqlalchemy import Column, String, Text, Float, DateTime, Boolean, Integer
# from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from config.database.session import Base


class NewsModel(Base):
    """뉴스 ORM 모델"""
    __tablename__ = "news"

    id = Column(String(50), primary_key=True)
    title = Column(String(500), nullable=False, index=True)
    content = Column(Text, nullable=False)  # 뉴스 내용 길이 길 수 있음
    source = Column(Text, nullable=False)  # dict(JSON) 저장 가능
    published_at = Column(DateTime, nullable=False, index=True)
    url = Column(String(1000))
    symbols = Column(String(500), index=True)  # 콤마로 구분된 심볼
    categories = Column(String(500))
    keywords = Column(Text)  # 길이 길어질 수 있음
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SentimentModel(Base):
    """감성 분석 ORM 모델"""
    __tablename__ = "sentiments"

    id = Column(String(50), primary_key=True)
    news_id = Column(String(50), nullable=False, index=True)
    score = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    keywords = Column(Text)
    reasoning = Column(Text)
    analyzed_at = Column(DateTime, default=datetime.utcnow)


class SubscriptionModel(Base):
    """구독 ORM 모델"""
    __tablename__ = "subscriptions"

    id = Column(String(50), primary_key=True)
    account_id = Column(String(100), nullable=False, index=True)
    symbols = Column(String(500), nullable=False)
    channels = Column(String(200), nullable=False)  # 콤마로 구분
    sentiment_threshold = Column(Float, default=0.5)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)