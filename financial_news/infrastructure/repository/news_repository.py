import json
from typing import List, Optional
from datetime import datetime, timedelta
from sqlalchemy import select, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession

from config.database.session import get_db_session, SessionLocal
from financial_news.application.port.output.news_repository_port import NewsRepositoryPort
from financial_news.domain.entity.news import News
from financial_news.domain.entity.sentiment import Sentiment
from financial_news.domain.value_objects.news_id import NewsId
from financial_news.domain.value_objects.sentiment_id import SentimentId
from financial_news.domain.value_objects.stock_symbol import StockSymbol
from financial_news.domain.value_objects.time_range import TimeRange
from financial_news.infrastructure.orm.models import NewsModel, SentimentModel


class NewsRepositoryImpl(NewsRepositoryPort):

    def __init__(self):
        self.db = SessionLocal()

    # 뉴스 저장
    def save(self, news: News) -> News:
        # source를 문자열로 변환 (dict면 JSON 직렬화)
        source_str = json.dumps(news.source) if isinstance(news.source, dict) else str(news.source)

        orm = NewsModel(
            id=str(news.id),
            title=news.title,
            content=news.content,
            source=source_str,
            published_at=news.published_at,
            url=news.url,
            symbols=",".join([str(s) for s in news.symbols]),
            categories=",".join(news.categories),
            keywords=",".join(news.keywords),
            created_at=news.created_at or datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        self.db.add(orm)
        self.db.commit()
        self.db.refresh(orm)

        return self._to_entity(orm)

    # ID 조회
    def find_by_id(self, news_id: NewsId) -> Optional[News]:
        orm = self.db.get(NewsModel, str(news_id))
        return self._to_entity(orm) if orm else None

    # 심볼 조회 (인터페이스와 100% 일치)
    def find_by_symbols(
        self,
        symbols: List[StockSymbol],
        time_range: Optional[TimeRange] = None,
        limit: int = 100
    ) -> List[News]:
        query = self.db.query(NewsModel)

        # 심볼 필터
        symbol_filters = [NewsModel.symbols.like(f"%{str(symbol)}%") for symbol in symbols]
        query = query.filter(or_(*symbol_filters))

        # 시간 범위 필터 안전하게 처리
        if isinstance(time_range, TimeRange):
            query = query.filter(
                NewsModel.published_at >= time_range.start,
                NewsModel.published_at <= time_range.end
            )

        # 정렬 + 제한
        orms = query.order_by(NewsModel.published_at.desc()).limit(limit).all()
        return [self._to_entity(o) for o in orms]

    # 최근 뉴스 조회 (파라미터 순서 일치)
    def find_recent(self, hours: int = 24, limit: int = 100) -> List[News]:
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        orms = (
            self.db.query(NewsModel)
            .filter(NewsModel.published_at >= cutoff_time)
            .order_by(NewsModel.published_at.desc())
            .limit(limit)
            .all()
        )

        return [self._to_entity(o) for o in orms]

    # 감성 저장
    def save_sentiment(self, sentiment: Sentiment) -> Sentiment:
        orm = SentimentModel(
            id=str(sentiment.id),
            news_id=str(sentiment.news_id),
            score=sentiment.score,
            confidence=sentiment.confidence,
            keywords=",".join(sentiment.keywords),
            reasoning=sentiment.reasoning,
            analyzed_at=sentiment.analyzed_at,
        )

        self.db.add(orm)
        self.db.commit()
        self.db.refresh(orm)

        return self._sentiment_to_entity(orm)

    # 감성 단건 조회 (이거 빠져서 계속 터졌던 거임)
    def find_sentiment_by_news_id(self, news_id: NewsId) -> Optional[Sentiment]:
        orm = (
            self.db.query(SentimentModel)
            .filter(SentimentModel.news_id == str(news_id))
            .first()
        )

        return self._sentiment_to_entity(orm) if orm else None

    # ORM → 도메인 변환
    def _to_entity(self, orm: NewsModel) -> News:
        # source를 dict로 복원 시도, 실패하면 문자열 그대로
        try:
            source = json.loads(orm.source)
        except (json.JSONDecodeError, TypeError):
            source = orm.source

        symbols = [StockSymbol(s) for s in orm.symbols.split(",") if s]
        categories = [c for c in orm.categories.split(",") if c]
        keywords = [k for k in orm.keywords.split(",") if k]

        return News(
            id=orm.id,
            title=orm.title,
            content=orm.content,
            source=source,
            published_at=orm.published_at,
            url=orm.url,
            symbols=symbols,
            categories=categories,
            keywords=keywords,
            created_at=orm.created_at,
            updated_at=orm.updated_at,
        )

    def _sentiment_to_entity(self, model: SentimentModel) -> Sentiment:
        return Sentiment(
            id=SentimentId(model.id),
            news_id=NewsId(model.news_id),
            score=model.score,
            confidence=model.confidence,
            keywords=model.keywords.split(",") if model.keywords else [],
            reasoning=model.reasoning,
            analyzed_at=model.analyzed_at,
        )