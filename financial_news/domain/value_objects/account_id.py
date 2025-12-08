from dataclasses import dataclass


@dataclass(frozen=True)
class AccountId:
    value: str

    @classmethod
    def from_string(cls, value: str) -> "AccountId":
        if not value or not value.strip():
            raise ValueError("Account ID cannot be empty")
        return cls(value=value.strip())

    def __str__(self) -> str:
        return self.value

    def __eq__(self, other) -> bool:
        if not isinstance(other, AccountId):
            return False
        return self.value == other.value

    def __hash__(self) -> int:
        return hash(self.value)