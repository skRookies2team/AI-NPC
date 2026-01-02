import logging
import os
from app.config import settings

def setup_logging() -> logging.Logger:
    """로깅 설정 및 logger 인스턴스 반환"""
    os.makedirs(settings.LOG_DIR, exist_ok=True)
    log_file = os.path.join(settings.LOG_DIR, settings.LOG_FILE)

    # 로거 설정
    logger = logging.getLogger("ai_server")
    logger.setLevel(logging.INFO)

    # 파일 핸들러 (로그 파일에 저장)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    # 콘솔 핸들러 (터미널에 출력)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 포맷터 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# 싱글톤 logger 인스턴스
logger = setup_logging()

def log_startup():
    """서버 시작 정보 로깅"""
    logger.info("=" * 50)
    logger.info("AI 서버 시작")
    logger.info("=" * 50)
    logger.info(f"S3_BUCKET: {settings.S3_BUCKET}")
    logger.info(f"AWS_REGION: {settings.AWS_REGION}")
    logger.info(f"AWS_ACCESS_KEY 설정됨: {bool(settings.AWS_ACCESS_KEY_ID)}")
    logger.info(f"AWS_SECRET_KEY 설정됨: {bool(settings.AWS_SECRET_ACCESS_KEY)}")
