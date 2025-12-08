import os
import aiohttp
import json
from typing import Dict, Any, Optional
from datetime import datetime

from dotenv import load_dotenv

from financial_news.application.port.output.notification_port import NotificationPort

load_dotenv()

class SlackNotificationAdapter(NotificationPort):
    """Slack 알림 어댑터"""

    def __init__(self):
        self.webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        if not self.webhook_url:
            print("WARNING: SLACK_WEBHOOK_URL not configured")

    async def send(
            self,
            channel: str,
            recipient: str,
            message: str,
            metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Slack 메시지 전송"""

        if not self.webhook_url:
            print("Slack webhook URL not configured, skipping notification")
            return False

        # Slack 메시지 포맷 생성
        slack_message = self._format_slack_message(message, metadata)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        self.webhook_url,
                        json=slack_message,
                        headers={"Content-Type": "application/json"},
                        timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        print(f"Slack notification sent successfully to {recipient}")
                        return True
                    else:
                        error_text = await response.text()
                        print(f"Slack notification failed: {response.status} - {error_text}")
                        return False

        except aiohttp.ClientError as e:
            print(f"Slack HTTP error: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error sending Slack notification: {e}")
            return False

    def _format_slack_message(
            self,
            message: str,
            metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Slack 메시지 포맷팅"""

        # 기본 메시지 구조
        slack_payload = {
            "text": message,
            "blocks": []
        }

        # 메타데이터가 있으면 Rich Formatting 적용
        if metadata:
            blocks = self._create_rich_blocks(message, metadata)
            slack_payload["blocks"] = blocks
        else:
            # 간단한 텍스트 메시지
            slack_payload["blocks"] = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": message
                    }
                }
            ]

        return slack_payload

    def _create_rich_blocks(
            self,
            message: str,
            metadata: Dict[str, Any]
    ) -> list:
        """Rich Formatting 블록 생성"""

        blocks = []

        # 헤더 블록
        symbol = metadata.get("symbol", "N/A")
        sentiment = metadata.get("sentiment", 0.0)

        # 감성에 따른 이모지
        emoji = self._get_sentiment_emoji(sentiment)

        blocks.append({
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{emoji} Financial News Alert: {symbol}",
                "emoji": True
            }
        })

        # 구분선
        blocks.append({"type": "divider"})

        # 메인 메시지
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": message
            }
        })

        # 메타데이터 필드
        fields = []

        if "symbol" in metadata:
            fields.append({
                "type": "mrkdwn",
                "text": f"*Symbol:*\n`{metadata['symbol']}`"
            })

        if "sentiment" in metadata:
            sentiment_label = self._get_sentiment_label(metadata['sentiment'])
            fields.append({
                "type": "mrkdwn",
                "text": f"*Sentiment:*\n{sentiment_label} ({metadata['sentiment']:.2f})"
            })

        if "news_count" in metadata:
            fields.append({
                "type": "mrkdwn",
                "text": f"*News Count:*\n{metadata['news_count']}"
            })

        if "timestamp" in metadata:
            fields.append({
                "type": "mrkdwn",
                "text": f"*Time:*\n{metadata['timestamp']}"
            })

        if fields:
            blocks.append({
                "type": "section",
                "fields": fields
            })

        # URL이 있으면 버튼 추가
        if "url" in metadata:
            blocks.append({
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "View Details",
                            "emoji": True
                        },
                        "url": metadata["url"],
                        "style": "primary"
                    }
                ]
            })

        # 푸터
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"📊 Financial News Analysis System | {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
                }
            ]
        })

        return blocks

    def _get_sentiment_emoji(self, sentiment: float) -> str:
        """감성 점수에 따른 이모지 반환"""
        if sentiment >= 0.6:
            return "🚀"
        elif sentiment >= 0.2:
            return "📈"
        elif sentiment >= -0.2:
            return "➡️"
        elif sentiment >= -0.6:
            return "📉"
        else:
            return "⚠️"

    def _get_sentiment_label(self, sentiment: float) -> str:
        """감성 점수 라벨"""
        if sentiment >= 0.6:
            return "🟢 Very Positive"
        elif sentiment >= 0.2:
            return "🟢 Positive"
        elif sentiment >= -0.2:
            return "🟡 Neutral"
        elif sentiment >= -0.6:
            return "🔴 Negative"
        else:
            return "🔴 Very Negative"


class SlackChannelNotificationAdapter(SlackNotificationAdapter):
    """특정 Slack 채널로 전송하는 어댑터"""

    def __init__(self, channel_webhook_url: str):
        """
        Args:
            channel_webhook_url: 특정 채널의 Webhook URL
        """
        super().__init__()
        self.webhook_url = channel_webhook_url