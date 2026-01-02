from datetime import datetime
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from app.core.logging import logger
import traceback

class LoggingMiddleware(BaseHTTPMiddleware):
    """요청/응답 로깅 미들웨어"""

    async def dispatch(self, request: Request, call_next):
        start_time = datetime.now()

        # 요청 정보 로깅
        logger.info(f"[REQUEST] {request.method} {request.url.path}")
        logger.debug(f"Query params: {dict(request.query_params)}")

        try:
            response = await call_next(request)
            process_time = (datetime.now() - start_time).total_seconds()

            # 응답 정보 로깅
            logger.info(
                f"[RESPONSE] {request.method} {request.url.path} - "
                f"Status: {response.status_code} - "
                f"Time: {process_time:.3f}s"
            )

            return response
        except Exception as e:
            process_time = (datetime.now() - start_time).total_seconds()
            logger.error(
                f"[ERROR] {request.method} {request.url.path} - "
                f"Error: {str(e)} - Time: {process_time:.3f}s"
            )
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
