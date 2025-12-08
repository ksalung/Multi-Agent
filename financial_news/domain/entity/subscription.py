from dataclasses import dataclass, field
from datetime import datetime
from typing import List

from financial_news.domain.entity.notification_channel import NotificationChannel
from financial_news.domain.value_objects.account_id import AccountId
from financial_news.domain.value_objects.stock_symbol import StockSymbol


@dataclass
class Subscription:
    """알림 구독 엔티티"""
    id: str
    account_id: AccountId
    symbols: List[StockSymbol]
    channels: List[NotificationChannel]
    sentiment_threshold: float = 0.5
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)

    def should_notify(self, sentiment_score: float) -> bool:
        """알림을 보내야 하는지 확인"""
        return self.is_active and abs(sentiment_score) >= self.sentiment_threshold

    def activate(self) -> None:
        """구독 활성화"""
        self.is_active = True

    def deactivate(self) -> None:
        """구독 비활성화"""
        self.is_active = False