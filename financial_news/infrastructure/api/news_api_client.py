import os

import aiohttp
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta


class NewsAPIClient:
    """NewsAPI.org 클라이언트"""

    BASE_URL = "https://newsapi.org/v2"

    def __init__(self):
        self.api_key = os.getenv("NEWS_API_KEY")
        if not self.api_key:
            print("WARNING: NEWS_API_KEY not configured")

    async def search_financial_news(
            self,
            symbols: List[str],
            from_date: Optional[datetime] = None,
            to_date: Optional[datetime] = None,
            page_size: int = 20
    ) -> List[Dict[str, Any]]:
        """금융 뉴스 검색"""

        if not self.api_key:
            return []

        # 심볼을 검색 쿼리로 변환
        query = " OR ".join(symbols)

        # 날짜 기본값
        if not from_date:
            from_date = datetime.utcnow() - timedelta(days=7)
        if not to_date:
            to_date = datetime.utcnow()

        params = {
            "q": query,
            "apiKey": self.api_key,
            "language": "en",
            "pageSize": page_size,
            "from": from_date.strftime("%Y-%m-%d"),
            "to": to_date.strftime("%Y-%m-%d"),
            "sortBy": "publishedAt",
            "domains": "bloomberg.com,reuters.com,cnbc.com,wsj.com"  # 금융 뉴스 사이트
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                        f"{self.BASE_URL}/everything",
                        params=params,
                        timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("articles", [])
                    else:
                        print(f"NewsAPI error: {response.status}")
                        return []

        except Exception as e:
            print(f"Failed to fetch from NewsAPI: {e}")
            return []