from account.application.port.account_repository_port import AccountRepositoryPort
from account.domain.account import Account


class AccountUseCase:
    def __init__(self, account_repository: AccountRepositoryPort):
        self.repo = account_repository

    def create_or_get_account(self, email: str, nickname: str | None):
        account = self.repo.find_by_email(email)
        if account:
            return account

        if not nickname:
            total = self.repo.count()
            nickname = f"anonymous{total + 1}"

        account = Account(email=email, nickname=nickname)
        return self.repo.save(account)
