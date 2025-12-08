from typing import List
from datetime import datetime

from financial_news.domain.entity.news import News
from financial_news.domain.value_objects.news_id import NewsId
from financial_news.infrastructure.api.rss_feed_client import RSSFeedClient


class RSSFeedAdapter:
    """RSS Feed 어댑터"""

    def __init__(self):
        self.rss_client = RSSFeedClient()

    async def fetch_latest_news(self, limit: int = 50) -> List[News]:
        """최신 뉴스 가져오기"""

        articles = await self.rss_client.fetch_all_feeds()

        news_list = []
        for article in articles[:limit]:
            try:
                news = News(
                    id=NewsId.generate(),
                    title=article.get("title", ""),
                    content=article.get("description", ""),
                    source=article.get("source", "RSS Feed"),
                    published_at=article.get("published_at", datetime.utcnow()),
                    url=article.get("url", ""),
                    symbols=[],
                    categories=["finance"],
                    keywords=[]
                )
                news_list.append(news)
            except Exception as e:
                print(f"Failed to convert RSS article: {e}")
                continue

        return news_list