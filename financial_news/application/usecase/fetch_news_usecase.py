from typing import List, Optional

from financial_news.adapter.output.google.news_api_adapter import NewsAPIAdapter
from financial_news.application.port.output.news_repository_port import NewsRepositoryPort
from financial_news.application.port.input.news_service_port import NewsServicePort
from financial_news.domain.entity.news import News
from financial_news.domain.value_objects.news_id import NewsId
from financial_news.domain.value_objects.stock_symbol import StockSymbol
from financial_news.domain.value_objects.time_range import TimeRange


class FetchNewsUseCase(NewsServicePort):
    """뉴스 조회 유스케이스"""

    def __init__(self, news_repository: NewsRepositoryPort):
        self.news_repository = news_repository
        self.news_api = NewsAPIAdapter()

    async def get_news_list(
            self,
            symbols: List[StockSymbol],
            limit: int = 20,
            time_range: Optional[TimeRange] = None
    ) -> List[News]:
        news_list = self.news_repository.find_by_symbols(symbols, limit)
        if news_list:
            return news_list

        articles = await self.news_api.fetch_news_by_symbols([str(s) for s in symbols], limit=limit)
        for news in articles:
            self.news_repository.save(news)

        return self.news_repository.find_by_symbols(symbols, limit)

    async def get_news_by_id(self, news_id: str) -> News | None:
        return self.news_repository.find_by_id(NewsId.from_string(news_id))
