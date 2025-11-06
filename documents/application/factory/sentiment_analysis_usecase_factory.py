from documents.adapter.output.ai.sentiment_agent import SentimentAgent
from documents.application.usecase.sentiment_analysis_usecase import SentimentAnalysisUseCase

class SentimentAnalysisUseCaseFactory:

    @staticmethod
    def create():
        sentiment_agent = SentimentAgent()
        return SentimentAnalysisUseCase(sentiment_agent)
