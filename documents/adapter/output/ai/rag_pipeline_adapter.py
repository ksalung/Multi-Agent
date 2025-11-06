import unicodedata
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from documents.domain.port.rag_port import RAGPort


# 텍스트 정제 유틸 (깨진 문자열 방지)
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)

    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if ch.isprintable())

    junk_patterns = ["œ", "‚", "˜", "”", "“", "’", "`", "´"]
    for j in junk_patterns:
        text = text.replace(j, "")

    return text.strip()


class RAGPipelineAdapter(RAGPort):
    def __init__(self, vector_db_adapter):
        self.vector_db = vector_db_adapter

        # Huggingface LLM 준비
        model_name = "facebook/bart-large-cnn"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        summarizer = pipeline(
            "summarization",
            model=model,
            tokenizer=tokenizer,
            max_length=256,
            min_length=32,
            do_sample=False
        )
        self.llm = HuggingFacePipeline(pipeline=summarizer)

        # retriever 준비 (여기서 output 타입이 다양한 경우 있음 → 정제 필요)
        self.retriever = vector_db_adapter.as_retriever()

        # Prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant. Use the provided context to answer the question."),
            ("human",
             "Context:\n{context}\n\n"
             "Question:\n{question}\n\n"
             "Provide a clear answer based only on the context above.")
        ])

        # Output parser
        self.output_parser = StrOutputParser()

        # LCEL chain 구성
        self.chain = (
            {
                "context": self._safe_retrieve,   # ⭐ retriever 안정화 래퍼
                "question": lambda x: x
            }
            | self.prompt
            | self.llm
            | self.output_parser
        )

    # retriever invoke 결과를 안전하게 문자열로 변환
    def _safe_retrieve(self, query: str):
        try:
            docs = self.retriever.invoke(query)

            # retriever가 list 반환 → 처리
            if isinstance(docs, list):
                contents = []
                for d in docs:
                    if hasattr(d, "page_content"):
                        contents.append(clean_text(d.page_content))
                    else:
                        contents.append(clean_text(str(d)))
                return "\n\n".join(contents)

            # retriever가 Document 단일 객체 반환
            if hasattr(docs, "page_content"):
                return clean_text(docs.page_content)

            # 기타 비정상 타입 → 문자열화
            return clean_text(str(docs))

        except Exception as e:
            # retriever 실패 시 fallback 메시지 제공
            return f"(Failed to retrieve documents: {e})"

    # core API
    def answer(self, query: str) -> str:
        response = self.chain.invoke(query)
        return clean_text(response)

    def query(self, query: str) -> str:
        return self.answer(query)
