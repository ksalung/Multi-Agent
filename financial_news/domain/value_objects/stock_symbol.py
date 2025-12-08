from dataclasses import dataclass
import re


@dataclass(frozen=True)
class StockSymbol:
    value: str

    def __post_init__(self):
        if not self._is_valid(self.value):
            raise ValueError(f"Invalid stock symbol: {self.value}")

    @staticmethod
    def _is_valid(symbol: str) -> bool:
        # 1~5자의 알파벳 대문자
        return bool(re.match(r'^[A-Z]{1,5}$', symbol))

    def __str__(self) -> str:
        return self.value

    def __eq__(self, other) -> bool:
        if not isinstance(other, StockSymbol):
            return False
        return self.value == other.value

    def __hash__(self) -> int:
        return hash(self.value)