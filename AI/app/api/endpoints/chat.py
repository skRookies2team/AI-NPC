from fastapi import HTTPException
from app.models.requests import ChatRequest
from app.services.chat import execute_chat
from app.core.logging import logger
import traceback

async def chat(request: ChatRequest):
    """채팅 메시지 처리"""
    logger.info(f"[CHAT] 시작 - session_id: {request.session_id}, character_name: {request.character_name}")
    logger.debug(f"사용자 메시지: {request.message[:100]}..." if len(request.message) > 100 else f"사용자 메시지: {request.message}")

    try:
        response = execute_chat(request.session_id, request.character_name, request.message)
        logger.info(f"[CHAT] 완료 - session_id: {request.session_id}")
        logger.debug(f"AI 응답: {response[:100]}..." if len(response) > 100 else f"AI 응답: {response}")
        return {"reply": response}

    except ValueError as e:
        logger.warning(f"[CHAT] 세션 없음: {e}")
        raise HTTPException(
            status_code=404,
            detail="Session not found or expired. 먼저 소설 학습(train-from-s3)과 캐릭터 설정(character)을 완료해주세요."
        )
    except Exception as e:
        logger.error(f"[CHAT] 오류 발생 - session_id: {request.session_id}, error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"채팅 처리 중 오류 발생: {str(e)}")
