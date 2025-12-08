from dataclasses import dataclass, field
from datetime import datetime

from financial_news.domain.value_objects.account_id import AccountId


@dataclass
class Account:
    """계정 엔티티"""
    id: AccountId
    email: str
    username: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_login: datetime = field(default_factory=datetime.utcnow)

    def update_last_login(self) -> None:
        """마지막 로그인 시간 업데이트"""
        self.last_login = datetime.utcnow()