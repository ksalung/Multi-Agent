from typing import List
import uuid

from financial_news.application.port.output.notification_port import NotificationPort
from financial_news.domain.entity.notification_channel import NotificationChannel
from financial_news.domain.entity.subscription import Subscription
from financial_news.domain.value_objects.account_id import AccountId
from financial_news.domain.value_objects.stock_symbol import StockSymbol


class SubscribeAlertUseCase:
    """알림 구독 유스케이스"""

    def __init__(self, notification_service: NotificationPort):
        self.notification_service = notification_service
        self.subscriptions = {}  # 실제로는 DB에 저장

    async def create_subscription(
            self,
            account_id: str,
            symbols: List[str],
            channels: List[str],
            sentiment_threshold: float = 0.5
    ) -> Subscription:
        """알림 구독 생성"""
        stock_symbols = [StockSymbol(s.upper()) for s in symbols]
        notification_channels = [NotificationChannel(c) for c in channels]

        subscription = Subscription(
            id=str(uuid.uuid4()),
            account_id=AccountId.from_string(account_id),
            symbols=stock_symbols,
            channels=notification_channels,
            sentiment_threshold=sentiment_threshold
        )

        self.subscriptions[subscription.id] = subscription

        return subscription

    async def get_subscriptions(self, account_id: str) -> List[Subscription]:
        """사용자의 구독 목록 조회"""
        user_account_id = AccountId.from_string(account_id)
        return [
            sub for sub in self.subscriptions.values()
            if sub.account_id == user_account_id
        ]

    async def send_alert(
            self,
            subscription: Subscription,
            news_title: str,
            symbol: str,
            sentiment_score: float
    ) -> bool:
        """알림 전송"""
        if not subscription.should_notify(sentiment_score):
            return False

        message = self._format_message(news_title, symbol, sentiment_score)

        success = True
        for channel in subscription.channels:
            try:
                await self.notification_service.send(
                    channel=channel.value,
                    recipient=str(subscription.account_id),
                    message=message,
                    metadata={
                        "symbol": symbol,
                        "sentiment": sentiment_score
                    }
                )
            except Exception as e:
                print(f"Failed to send notification via {channel}: {e}")
                success = False

        return success

    def _format_message(
            self,
            news_title: str,
            symbol: str,
            sentiment_score: float
    ) -> str:
        """알림 메시지 포맷팅"""
        sentiment_label = "POSITIVE" if sentiment_score > 0 else "NEGATIVE"

        return f"""
Financial News Alert

Symbol: {symbol}
Sentiment: {sentiment_label} ({sentiment_score:.2f})

{news_title}
        """.strip()