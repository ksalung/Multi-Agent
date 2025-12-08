from dataclasses import dataclass
from uuid import uuid4


@dataclass(frozen=True)
class NewsId:
    value: str

    @classmethod
    def generate(cls) -> "NewsId":
        return cls(value=str(uuid4()))

    @classmethod
    def from_string(cls, value: str) -> "NewsId":
        return cls(value=value)

    def __str__(self) -> str:
        return self.value

    def __eq__(self, other) -> bool:
        if not isinstance(other, NewsId):
            return False
        return self.value == other.value

    def __hash__(self) -> int:
        return hash(self.value)