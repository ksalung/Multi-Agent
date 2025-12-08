from abc import ABC, abstractmethod
from typing import Dict, Any


class NotificationPort(ABC):
    """알림 서비스 포트 (Output Port)"""

    @abstractmethod
    async def send(
            self,
            channel: str,
            recipient: str,
            message: str,
            metadata: Dict[str, Any] = None
    ) -> bool:
        """알림 전송"""
        pass