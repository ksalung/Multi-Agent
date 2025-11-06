# sentiment_agent.py
from transformers import pipeline
from documents.domain.port.sentiment_analysis_port import SentimentAnalysisPort


class SentimentAgent(SentimentAnalysisPort):
    """
    HuggingFace transformers 기반 감성 분석 에이전트.
    DDD 구조에 맞게 Port(SentimentAnalysisPort)를 구현한다.
    """

    def __init__(self, model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment"):
        """
        기본 sentiment-analysis 모델은 영어 중심이므로
        한국어/다국어 지원 모델로 설정함.
        """
        try:
            self.analyzer = pipeline("sentiment-analysis", model=model_name)
        except Exception:
            # fallback: transformers 기본 모델
            self.analyzer = pipeline("sentiment-analysis")

    def analyze(self, text: str) -> dict:
        """
        긴 텍스트 감성 분석 시에는 transformers 오류 방지를 위해
        일정 길이로 슬라이스 후 분석한다.
        """
        if not text:
            return {"label": "unknown", "score": 0.0}

        safe_text = text[:3000]  # 매우 긴 텍스트 보호

        try:
            result = self.analyzer(safe_text)
        except Exception:
            # 모델 오류 발생 시 간단한 fallback
            return {"label": "error", "score": 0.0}

        # transformers pipeline은 리스트 형태로 반환됨
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
            return result[0]

        return {"label": "unknown", "score": 0.0}
