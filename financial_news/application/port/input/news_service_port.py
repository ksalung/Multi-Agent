from abc import ABC, abstractmethod
from typing import List, Optional

from financial_news.domain.entity.news import News
from financial_news.domain.value_objects.time_range import TimeRange


class NewsServicePort(ABC):
    """뉴스 서비스 포트 (Input Port)"""

    @abstractmethod
    async def get_news_list(
            self,
            symbols: Optional[List[str]] = None,
            time_range: Optional[TimeRange] = None,
            limit: int = 20
    ) -> List[News]:
        """뉴스 목록 조회"""
        pass

    @abstractmethod
    async def get_news_by_id(self, news_id: str) -> Optional[News]:
        """뉴스 상세 조회"""
        pass