"""
AI NPC Server - Main Entry Point

리팩토링된 모듈 구조의 진입점입니다.
기존 ai_server.py의 모든 기능을 유지하면서 코드가 여러 모듈로 분리되었습니다.

사용법:
    python ai_server.py
"""

from app.main import app
from app.core.logging import logger
from app.config import settings

if __name__ == "__main__":
    import uvicorn
    logger.info(f"서버 시작 - 포트: {settings.PORT}")
    # AI 서버 포트: 8002
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
