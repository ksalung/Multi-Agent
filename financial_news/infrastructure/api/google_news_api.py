import os
from typing import List, Optional
from datetime import datetime
from serpapi import GoogleSearch
import asyncio
from concurrent.futures import ThreadPoolExecutor

class GoogleNewsAPIClient:
    """SERP API 기반 뉴스 API 호출 (dict 반환)"""

    def __init__(self):
        self.api_key = os.getenv("SERP_API_KEY")
        if not self.api_key:
            raise ValueError("SERP API key not configured")

    async def fetch_news(self, query: str, limit: int = 20,
                         from_date: Optional[datetime] = None,
                         to_date: Optional[datetime] = None) -> List[dict]:

        def _search():
            params = {"engine": "google_news", "q": query, "api_key": self.api_key, "num": limit}
            if from_date:
                params["from"] = from_date.isoformat()
            if to_date:
                params["to"] = to_date.isoformat()
            search = GoogleSearch(params)
            return search.get_dict().get("news_results", [])

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(ThreadPoolExecutor(), _search)
