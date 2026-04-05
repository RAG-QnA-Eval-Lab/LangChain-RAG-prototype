# server/utils/chain.py
import os
import logging
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import config

from .embedding import SimpleLocalEmbeddings
from .llm import HuggingFaceInferenceAPI
from .document import check_faiss_index_exists

logger = logging.getLogger(__name__)

# RAG 파이프라인을 초기화
def initialize_rag_pipeline() -> RetrievalQA:
    try:
        # 로컬 임베딩 모델 초기화 (문서 임베딩은 이미 생성된 index 사용)
        embeddings = SimpleLocalEmbeddings()

        # FAISS 인덱스 확인
        if check_faiss_index_exists():
            # 기존 FAISS 인덱스 로드
            logger.info(f"기존 FAISS 인덱스 로드: {config.FAISS_INDEX_PATH}")
            # SECURITY: allow_dangerous_deserialization=True는 내부에서 생성한
            # FAISS 인덱스 파일만 사용하므로 허용. 외부 출처의 파일은 절대 사용 금지.
            vectorstore = FAISS.load_local(
                folder_path=str(config.DATA_DIR),
                embeddings=embeddings,
                index_name="document_index",
                allow_dangerous_deserialization=True
            )
        else:
            logger.error(f"FAISS 인덱스를 찾을 수 없습니다: {config.FAISS_INDEX_PATH}")
            raise FileNotFoundError(f"FAISS 인덱스 파일이 존재하지 않습니다: {config.FAISS_INDEX_PATH}")
        
        # retriever 생성
        retriever = vectorstore.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance - 다양성과 관련성 균형
            search_kwargs={
                "k": 5,  # 최종 반환 문서 수
                "fetch_k": 40,  # 초기 검색 문서 수
                "lambda_mult": 0.8,  # 관련성 가중치 (0~1)
            }
        )
        
        # LLM 초기화 
        llm = HuggingFaceInferenceAPI(
            temperature=0.2,
            max_tokens=512,
            top_p=0.95,
            model_name="meta-llama/Llama-3.1-8B-Instruct"
        )
        
        prompt_template = """다음 여러 문서의 내용을 바탕으로 질문에 답변하세요.
        
문서 내용:
{context}

질문: {question}

최종 답변 (모든 정보를 종합하여 간결하게 한 문단으로 작성하세요. 문장 사이에는 공백만 사용하고, 절대 줄바꿈 문자(\\n)를 포함하지 마세요. 문서에 관련 정보가 없으면 "주어진 문맥에서 답를 찾을 수 없습니다."라고만 응답):"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # QA 체인 생성 
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        logger.info("RAG 파이프라인이 성공적으로 초기화되었습니다.")
        return qa_chain
    
    except Exception as e:
        logger.error(f"RAG 파이프라인 초기화 오류: {str(e)}")
        raise RuntimeError(f"RAG 파이프라인 초기화 중 오류가 발생했습니다: {str(e)}")
