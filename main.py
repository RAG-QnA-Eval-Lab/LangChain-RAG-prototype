# main.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from server.routers.question_answer import router as question_answer_router
import config

from pathlib import Path
from dotenv import load_dotenv
import json
import uvicorn
import logging
import logging.config
import traceback
import os

# Load environment variables
load_dotenv()

# Configure logging from JSON file
with open(config.LOG_CONFIG_PATH, "r") as f:
    logging_config = json.load(f)

logging.config.dictConfig(logging_config)
logger = logging.getLogger("server")

# Initialize FastAPI app
app = FastAPI(title="LLM & RAG Service with wikipedia data")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

# Include routers
app.include_router(question_answer_router, prefix="/api/llmserver", tags=["Q&A"])

# Root endpoint
@app.get("/")
def read_root():
    logger.info("Root endpoint accessed")
    return {"message": "Hello, Toby!"}

@app.get("/health")
async def health():
    return {"status": "ok"}

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.debug("Request received")
    try:
        response = await call_next(request)
        logger.info("Response sent", extra={
            "errorType": "",
            "error_message": ""
        })
        return response
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        if tb:
            filename, lineno, func, _ = tb[-1]
        else:
            filename, lineno, func = ("unknown", 0, "unknown")
        logger.error("Error processing request", extra={
            "errorType": type(e).__name__,
            "error_message": str(e)
        })
        raise

# Exception handler: catch-all HTTPException
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error("HTTP exception", extra={
        "errorType": type(exc).__name__,
        "error_message": str(exc.detail)
    })
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

# Exception handler: catch-all
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception", extra={
        "errorType": type(exc).__name__,
        "error_message": str(exc)
    })
    return JSONResponse(
        status_code=500,
        content={"detail": "서버 내부 오류가 발생했습니다."},
    )

# Test logging endpoint
@app.get("/test-logging")
def test_logging():
    logger.debug("디버그 레벨 로그 테스트")
    logger.info("정보 레벨 로그 테스트")
    logger.error("오류 레벨 로그 테스트")
    return {"message": "로깅 테스트 완료"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_config="logging_config.json"
    )

# uvicorn main:app --host 0.0.0.0 --port 8000 --reload --log-config logging_config.json
