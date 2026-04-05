# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KorQuAD 1.0 (Wikipedia) 데이터 기반 RAG (Retrieval-Augmented Generation) Q&A 서버. FastAPI + LangChain + FAISS + HuggingFace Inference API 구성.

## Commands

```bash
# 서버 실행
python main.py
# 또는
uvicorn main:app --host 0.0.0.0 --port 8000 --reload --log-config logging_config.json

# Docker
docker build -t langchain-rag .
docker run -p 8000:8000 langchain-rag

# 의존성 설치
pip install -r requirements.txt

# 데이터 전처리 (Jupyter notebook)
# data_preprocess/ 노트북 실행 후 생성된 .faiss, .pkl 파일을 data/ 에 배치
```

테스트 프레임워크는 아직 미구성 상태.

## Architecture

```
main.py                         # FastAPI 앱 초기화, 미들웨어, 예외 핸들러
├── config.py                   # 환경 감지(Docker/로컬), 경로 상수, 디렉토리 생성
└── server/
    ├── routers/question_answer.py  # POST /api/llmserver/query 엔드포인트
    ├── schemas/schema.py           # Pydantic 모델 (Question, AnswerResponse)
    └── utils/
        ├── chain.py       # RAG 파이프라인 조립 (RetrievalQA)
        ├── document.py    # 문서/인덱스 로드, FAISS 존재 확인
        ├── embedding.py   # OpenAI Embedding 래퍼 (text-embedding-ada-002)
        └── llm.py         # HuggingFace Inference API LLM 래퍼 (Llama-3.1-8B-Instruct)
```

### Request Flow

1. 클라이언트 → `POST /api/llmserver/query` (question 전송)
2. `RAGPipelineSingleton`이 최초 요청 시 파이프라인 초기화 (싱글톤)
3. OpenAI API로 질문 임베딩 → FAISS에서 MMR 기반 유사 문서 검색 (k=5, fetch_k=40)
4. 검색된 문서 + 질문을 프롬프트로 조합 → HuggingFace Inference API (Llama-3.1-8B-Instruct)로 답변 생성
5. 답변 + 검색 문서를 JSON 응답으로 반환

### Key Patterns

- **싱글톤**: `RAGPipelineSingleton` — RAG 파이프라인을 한 번만 초기화하고 FastAPI `Depends`로 주입
- **LangChain 커스텀 래퍼**: `SimpleLocalEmbeddings(Embeddings)`, `HuggingFaceInferenceAPI(LLM)` — LangChain 베이스 클래스를 상속하여 외부 API 연동
- **환경 자동 감지**: `config.py`에서 `/proc/1/cgroup` 읽어 Docker/로컬 환경 자동 판별 후 경로 설정

## Environment Variables

`.env` 파일에 설정 (필수):
- `OPENAI_API_KEY` — 쿼리 임베딩용
- `HUGGINGFACE_API_KEY` — LLM 추론용 (provider: novita)

## Data Files

`data/` 디렉토리에 전처리된 파일 필요:
- `documents.json` — 전처리된 KorQuAD 위키피디아 문서
- `document_index.faiss` — FAISS 벡터 인덱스
- `document_index.pkl` — FAISS 메타데이터 (pickle, 내부 생성 파일만 사용)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/llmserver/query` | POST | `{"question": "..."}` → 답변 + 검색 문서 반환 |
| `/health` | GET | 서버 상태 확인 |
| `/docs` | GET | Swagger UI |
