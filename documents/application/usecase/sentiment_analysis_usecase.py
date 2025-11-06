from documents.domain.port.sentiment_analysis_port import SentimentAnalysisPort

class SentimentAnalysisUseCase:

    def __init__(self, sentiment_agent: SentimentAnalysisPort):
        self.sentiment_agent = sentiment_agent

    async def execute(self, text: str) -> dict:
        return self.sentiment_agent.analyze(text)
