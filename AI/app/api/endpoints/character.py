from fastapi import HTTPException
from app.models.requests import CharacterRequest
from app.services.character import set_character_info
from app.core.logging import logger
import traceback

async def set_character(request: CharacterRequest):
    """
    캐릭터 정보를 입력받아 시스템 프롬프트 설정
    소설 학습과 별도로 캐릭터 정보만 업데이트할 수 있음
    PostgreSQL에 벡터 스토어가 이미 존재하면 로드하여 대화 준비 완료
    """
    logger.info(f"[CHARACTER] 시작 - session_id: {request.session_id}, character_name: {request.character_name}")

    try:
        system_prompt, ready_for_chat = set_character_info(
            request.session_id,
            request.character_name,
            request.character_description
        )

        logger.info(f"[CHARACTER] 완료 - session_id: {request.session_id}, character_name: {request.character_name}")

        return {
            "status": "character_set",
            "session_id": request.session_id,
            "character_name": request.character_name,
            "ready_for_chat": ready_for_chat,
            "message": "캐릭터 정보가 성공적으로 설정되었습니다." +
                      (" 대화할 준비가 완료되었습니다." if ready_for_chat else
                       " 소설 학습(train-from-s3)이 완료되면 대화할 수 있습니다.")
        }

    except Exception as e:
        logger.error(f"[CHARACTER] 오류 발생 - session_id: {request.session_id}, error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"캐릭터 정보 설정 중 오류 발생: {str(e)}")
