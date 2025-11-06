from abc import ABC, abstractmethod

class SentimentAnalysisPort(ABC):
    @abstractmethod
    def analyze_sentiment(self, text: str): ...
