from abc import ABC, abstractmethod
from typing import List, Dict, Any

from financial_news.domain.entity.analysis_report import AnalysisReport


class AnalysisServicePort(ABC):
    """분석 서비스 포트 (Input Port)"""

    @abstractmethod
    async def generate_report(self, symbols: List[str], days: int = 7) -> AnalysisReport:
        """분석 리포트 생성"""
        pass

    @abstractmethod
    async def get_trending_topics(self, limit: int = 10) -> List[Dict[str, Any]]:
        """트렌딩 토픽 조회"""
        pass