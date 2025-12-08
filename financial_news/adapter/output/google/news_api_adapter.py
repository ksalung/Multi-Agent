from typing import List
from datetime import datetime
from dateutil import parser
from financial_news.domain.entity.news import News
from financial_news.domain.value_objects.news_id import NewsId
from financial_news.domain.value_objects.stock_symbol import StockSymbol
from financial_news.infrastructure.api.google_news_api import GoogleNewsAPIClient

class NewsAPIAdapter:
    """Google News API Adapter → 도메인 엔티티 변환"""

    def __init__(self):
        self.google_client = GoogleNewsAPIClient()

    async def fetch_news_by_symbols(self, symbols: List[str], limit: int = 20) -> List[News]:
        # query 생성
        query = "stock market " + " OR ".join(symbols)
        articles = await self.google_client.fetch_news(query=query, limit=limit)

        news_list = []
        for article in articles[:limit]:
            news_list.append(self._convert_to_domain_entity(article, symbols))

        return news_list

    def _parse_published_at(self, date_str: str) -> datetime:
        """Google News API 날짜 문자열 안전하게 파싱"""
        if not date_str:
            return datetime.utcnow()
        try:
            # dateutil 시도
            return parser.parse(date_str)
        except (ValueError, TypeError):
            # Google News API 포맷 예외 처리
            try:
                # 예: 12/04/2025, 01:30 PM, +0000 UTC
                return datetime.strptime(date_str, "%m/%d/%Y, %I:%M %p, %z UTC")
            except ValueError:
                # 마지막 fallback
                return datetime.utcnow()

    def _convert_to_domain_entity(self, article: dict, symbols: List[str]) -> News:
        content = f"{article.get('title', '')} {article.get('snippet', '')}"
        detected_symbols = [StockSymbol(s.upper()) for s in symbols if s.upper() in content.upper()]

        published_at = article.get("date") or article.get("published_at")
        if isinstance(published_at, str):
            published_at = self._parse_published_at(published_at)
        else:
            published_at = datetime.utcnow()

        return News(
            id=NewsId.generate(),
            title=article.get("title", ""),
            content=article.get("snippet", ""),
            source=article.get("source", "Unknown"),
            published_at=published_at,
            url=article.get("link", ""),
            symbols=detected_symbols,
            categories=["finance", "stock"],
            keywords=[]
        )
