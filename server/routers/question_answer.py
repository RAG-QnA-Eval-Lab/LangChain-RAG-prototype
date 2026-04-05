# server/routers/question_answer.py
import logging
from fastapi import APIRouter, HTTPException, Depends
import time

from server.schemas.schema import Question, AnswerResponse
from server.utils import initialize_rag_pipeline

# 싱글톤 패턴으로 RAG 파이프라인 관리
class RAGPipelineSingleton:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = initialize_rag_pipeline()
        return cls._instance

# 의존성 주입을 위한 함수
def get_rag_pipeline():
    return RAGPipelineSingleton.get_instance()

router = APIRouter()
logger = logging.getLogger(__name__)

# 질문에 대한 답변 생성
@router.post("/query", response_model=AnswerResponse)
async def answer_question(request: Question, rag_pipeline=Depends(get_rag_pipeline)):
    try:
        start_time = time.time()
        query = request.question
        logger.info(f"답변 생성 요청: '{query}'")
        
        # RAG 파이프라인 실행
        result = rag_pipeline.invoke({"query": query})
        
        # 결과 추출
        answer = result["result"]
        source_docs = result.get("source_documents", [])
        
        # 검색된 문서가 없는 경우
        if not source_docs:
            logger.warning(f"검색된 문서 없음: '{query}'")
            raise HTTPException(status_code=404, detail="관련 문서를 찾을 수 없습니다.")
        
        # 가장 관련성 높은 문서 사용
        best_doc = source_docs[0]
        doc_content = best_doc.page_content
        doc_metadata = best_doc.metadata
        
        # 문서 ID 가져오기 - 숫자로 변환
        try:
            doc_id = int(doc_metadata.get("id").split("-")[0])
        except (ValueError, AttributeError, IndexError):
            doc_id = 1  # 기본값
        
        # 응답 생성
        response = AnswerResponse(
            retrieved_document_id=doc_id,
            retrieved_document=doc_content,
            question=query,
            answers=answer
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"답변 생성 완료: 처리 시간={elapsed_time:.2f}초")
        
        return response
    
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        
        # 오류 정보 자세히 로깅
        logger.error(f"답변 생성 오류: {str(e) if str(e) else '(빈 오류 메시지)'}")
        logger.error(f"오류 타입: {type(e).__name__}")
        logger.error(f"스택 트레이스: {error_traceback}")
        
        # HTTP 예외 발생
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"답변 생성 중 오류가 발생했습니다: {str(e) if str(e) else '알 수 없는 오류'}")