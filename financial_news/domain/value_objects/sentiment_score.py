from dataclasses import dataclass

from financial_news.domain.value_objects.sentiment_label import SentimentLabel


@dataclass(frozen=True)
class SentimentScore:
    value: float

    def __post_init__(self):
        if not -1.0 <= self.value <= 1.0:
            raise ValueError(f"Sentiment score must be between -1.0 and 1.0, got {self.value}")

    def get_label(self) -> SentimentLabel:
        if self.value <= -0.6:
            return SentimentLabel.VERY_NEGATIVE
        elif self.value <= -0.2:
            return SentimentLabel.NEGATIVE
        elif self.value <= 0.2:
            return SentimentLabel.NEUTRAL
        elif self.value <= 0.6:
            return SentimentLabel.POSITIVE
        else:
            return SentimentLabel.VERY_POSITIVE

    def is_positive(self) -> bool:
        return self.value > 0.2

    def is_negative(self) -> bool:
        return self.value < -0.2