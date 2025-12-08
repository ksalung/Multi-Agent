import json
from typing import List
from openai import AsyncOpenAI
from config.openai.config import get_openai_config
from financial_news.application.port.output.ai_service_port import AIServicePort, SentimentAnalysisResult


class OpenAISentimentAdapter(AIServicePort):
    """OpenAI 감성 분석 어댑터"""

    def __init__(self):
        openai_config = get_openai_config()
        self.client = AsyncOpenAI(api_key=openai_config.api_key)
        self.model = openai_config.model

    async def analyze_sentiment(self, title: str, content: str) -> SentimentAnalysisResult:
        """텍스트 감성 분석"""
        prompt = f"""
Analyze the sentiment of the following financial news article.

Title: {title}
Content: {content[:1000]}

Provide your analysis in the following JSON format:
{{
    "score": <float between -1.0 and 1.0>,
    "confidence": <float between 0.0 and 1.0>,
    "keywords": [<list of important financial keywords>],
    "reasoning": "<brief explanation of your analysis>"
}}

Rules:
- score: -1.0 (very negative) to 1.0 (very positive)
- confidence: how confident you are in this analysis
- keywords: extract 5-10 important financial terms
- reasoning: explain why you gave this score
"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial sentiment analysis expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )

            content_str = response.choices[0].message['content']
            result = json.loads(content_str)

            return SentimentAnalysisResult(
                score=float(result['score']),
                confidence=float(result['confidence']),
                keywords=result['keywords'],
                reasoning=result['reasoning']
            )
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return SentimentAnalysisResult(
                score=0.0,
                confidence=0.0,
                keywords=[],
                reasoning="Failed to analyze sentiment"
            )

    async def extract_keywords(self, text: str, limit: int = 10) -> List[str]:
        """키워드 추출"""
        prompt = f"""
Extract the {limit} most important financial keywords from the following text:

{text[:1000]}

Return only a JSON object with a keywords array:
{{"keywords": ["keyword1", "keyword2", ...]}}
"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )

            content_str = response.choices[0].message['content']
            result = json.loads(content_str)
            return result.get('keywords', [])
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return []

    async def summarize(self, text: str, max_length: int = 200) -> str:
        """텍스트 요약"""
        prompt = f"""
Summarize the following financial news in {max_length} characters or less:

{text[:2000]}

Provide a concise summary focusing on key financial information.
"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150
            )

            content_str = response.choices[0].message['content']
            return content_str.strip()
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return "Failed to generate summary"
