# config.py
from dotenv import load_dotenv
from pathlib import Path
import os
import logging

# 모듈별 로거 생성
logger = logging.getLogger(__name__)

# .env 파일에서 환경 변수 로드
load_dotenv()

# Docker 환경 감지 및 경로 설정
try:
    with open('/proc/1/cgroup', 'rt') as f:
        cgroup_content = f.read()
    if 'docker' in cgroup_content:
        # Docker 환경
        BASE_DIR = Path("/app")
        logger.info("Docker 환경으로 감지되었습니다.")
        logger.debug(f"BASE_DIR 설정: {BASE_DIR}")
    else:
        # 로컬 환경
        BASE_DIR = Path(__file__).resolve().parent
        logger.info("로컬 환경으로 감지되었습니다.")
        logger.debug(f"BASE_DIR 설정: {BASE_DIR}")
except FileNotFoundError:
    # 로컬 환경
    BASE_DIR = Path(__file__).resolve().parent
    logger.info("로컬 환경으로 간주합니다. 기본 경로로 설정합니다.")
    logger.debug(f"BASE_DIR 설정: {BASE_DIR}")
except Exception as e:
    logger.error(f"환경 감지 중 오류 발생: {e}", extra={
        "errorType": "EnvironmentDetectionError",
        "error_message": str(e)
    })
    raise RuntimeError("환경 감지 중 오류 발생") from e

# 데이터 관련 경로
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_PATH = DATA_DIR / "documents.json"
FAISS_INDEX_PATH = DATA_DIR / "document_index.faiss"
ID_TO_INDEX_PATH = DATA_DIR / "id_to_index.pkl"

# 로그 설정
LOGS_DIR = BASE_DIR / "logs"
LOG_CONFIG_PATH = BASE_DIR / "logging_config.json"

# 디렉토리 존재 여부 확인 및 생성
try:
    for directory in [DATA_DIR, LOGS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"디렉토리가 준비되었습니다: {directory}")
        logger.debug(f"생성된 디렉토리 경로: {directory}")
except Exception as e:
    logger.error(f"디렉토리 생성 실패: {e}", extra={
        "errorType": "DirectoryCreationError",
        "error_message": str(e)
    })
    raise RuntimeError("디렉토리 생성 실패") from e