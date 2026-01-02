from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.requests import Request as StarletteRequest
from fastapi.responses import JSONResponse

from app.config import settings
from app.core.logging import logger, log_startup
from app.database.session import init_session_table
from app.middleware.logging import LoggingMiddleware
from app.api.routes import register_routes

def create_app() -> FastAPI:
    """FastAPI 애플리케이션 생성 및 설정"""

    # 앱 생성
    app = FastAPI()

    # 요청 검증 오류 핸들러 추가
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: StarletteRequest, exc: RequestValidationError):
        """요청 검증 오류 상세 로깅"""
        errors = exc.errors()
        error_details = []
        for error in errors:
            field = " -> ".join(str(loc) for loc in error["loc"])
            error_details.append({
                "field": field,
                "message": error["msg"],
                "type": error["type"],
                "input": error.get("input")
            })

        # 요청 본문 읽기 시도
        body_str = "N/A"
        try:
            body_bytes = await request.body()
            if body_bytes:
                body_str = body_bytes.decode('utf-8')[:500]  # 처음 500자만
        except Exception as e:
            body_str = f"요청 본문 읽기 실패: {str(e)}"

        logger.error(f"[VALIDATION ERROR] {request.method} {request.url.path}")
        logger.error(f"[VALIDATION ERROR] 오류 상세: {error_details}")
        logger.error(f"[VALIDATION ERROR] 요청 본문: {body_str}")
        logger.error(f"[VALIDATION ERROR] 요청 헤더: {dict(request.headers)}")

        return JSONResponse(
            status_code=422,
            content={
                "detail": error_details,
                "message": "요청 형식이 올바르지 않습니다."
            }
        )

    # 미들웨어 추가
    app.add_middleware(LoggingMiddleware)

    # 라우트 등록
    register_routes(app)

    # 시작 이벤트
    @app.on_event("startup")
    async def startup_event():
        """애플리케이션 시작 시 초기화"""
        log_startup()
        init_session_table()
        logger.info("애플리케이션 시작 완료")

    return app

# 앱 인스턴스 생성
app = create_app()
