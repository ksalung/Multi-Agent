from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Tuple


@dataclass(frozen=True)
class TimeRange:
    start: datetime
    end: datetime

    def __post_init__(self):
        if self.start >= self.end:
            raise ValueError("Start time must be before end time")

    @classmethod
    def last_n_days(cls, days: int) -> "TimeRange":
        end = datetime.utcnow()
        start = end - timedelta(days=days)
        return cls(start=start, end=end)

    @classmethod
    def today(cls) -> "TimeRange":
        now = datetime.utcnow()
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return cls(start=start, end=now)

    def duration_hours(self) -> float:
        return (self.end - self.start).total_seconds() / 3600

    def to_tuple(self) -> Tuple[datetime, datetime]:
        return (self.start, self.end)