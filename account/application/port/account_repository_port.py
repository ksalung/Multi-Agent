from typing import Optional
from abc import ABC, abstractmethod
from account.domain.account import Account

class AccountRepositoryPort(ABC):

    @abstractmethod
    def save(self, account: Account) -> Account:
        pass

    @abstractmethod
    def find_by_email(self, email: str) -> Optional[Account]:
        pass

    @abstractmethod
    def count(self) -> int:
        pass
