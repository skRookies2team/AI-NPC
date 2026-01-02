from fastapi import HTTPException
from app.database.session import list_all_sessions, load_all_characters_for_session
from app.core.logging import logger
import traceback

async def list_sessions():
    """
    모든 세션(게임) 목록 조회
    사용자가 만든 모든 소설 게임 목록을 반환
    """
    logger.info("[LIST-SESSIONS] 세션 목록 조회 요청")
    try:
        sessions = list_all_sessions()
        return {
            "status": "success",
            "sessions": sessions,
            "count": len(sessions)
        }
    except Exception as e:
        logger.error(f"[LIST-SESSIONS] 오류 발생 - error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"세션 목록 조회 중 오류 발생: {str(e)}")

async def get_session_info(session_id: str):
    """
    특정 세션의 상세 정보 조회 (모든 캐릭터 포함)
    게임을 시작하기 전에 세션 정보를 확인할 수 있음
    """
    logger.info(f"[GET-SESSION] 세션 정보 조회 요청 - session_id: {session_id}")
    try:
        characters = load_all_characters_for_session(session_id)
        if not characters:
            raise HTTPException(status_code=404, detail="Session not found")

        # system_prompt는 제외하고 반환 (너무 길 수 있음)
        characters_info = []
        for char in characters:
            characters_info.append({
                "character_name": char["character_name"],
                "character_description": char.get("character_description"),
                "created_at": str(char.get("created_at")),
                "updated_at": str(char.get("updated_at"))
            })

        return {
            "status": "success",
            "session_id": session_id,
            "characters": characters_info,
            "character_count": len(characters_info)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[GET-SESSION] 오류 발생 - session_id: {session_id}, error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"세션 정보 조회 중 오류 발생: {str(e)}")
