import aiohttp
from typing import List, Dict, Any
from datetime import datetime


class RSSFeedClient:
    """RSS Feed 파서 클라이언트"""

    # 주요 금융 뉴스 RSS 피드
    FINANCIAL_RSS_FEEDS = [
        "https://www.reuters.com/business/finance/rss",
        "https://feeds.bloomberg.com/markets/news.rss",
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "https://www.wsj.com/xml/rss/3_7085.xml"
    ]

    async def fetch_from_feed(self, feed_url: str) -> List[Dict[str, Any]]:
        """단일 RSS 피드에서 뉴스 가져오기"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                        feed_url,
                        timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        content = await response.text()
                        feed = feedparser.parse(content)

                        articles = []
                        for entry in feed.entries[:20]:  # 최대 20개
                            article = {
                                "title": entry.get("title", ""),
                                "description": entry.get("summary", ""),
                                "url": entry.get("link", ""),
                                "published_at": self._parse_date(entry.get("published")),
                                "source": feed.feed.get("title", "Unknown")
                            }
                            articles.append(article)

                        return articles
                    else:
                        return []

        except Exception as e:
            print(f"Failed to fetch RSS feed {feed_url}: {e}")
            return []

    async def fetch_all_feeds(self) -> List[Dict[str, Any]]:
        """모든 RSS 피드에서 뉴스 가져오기"""
        all_articles = []

        for feed_url in self.FINANCIAL_RSS_FEEDS:
            articles = await self.fetch_from_feed(feed_url)
            all_articles.extend(articles)

        # 중복 제거 (URL 기준)
        seen_urls = set()
        unique_articles = []
        for article in all_articles:
            if article["url"] not in seen_urls:
                seen_urls.add(article["url"])
                unique_articles.append(article)

        return unique_articles

    def _parse_date(self, date_str: str) -> datetime:
        """날짜 문자열 파싱"""
        try:
            from dateutil import parser
            return parser.parse(date_str)
        except:
            return datetime.utcnow()