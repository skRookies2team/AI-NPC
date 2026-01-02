from typing import Dict, Optional

# 전역 상태 관리 (인메모리 캐시)
# 서버 재시작 시 초기화되지만, 요청 간에는 유지됨
vector_store_mapping: Dict[str, any] = {}  # session_id -> retriever
system_prompts: Dict[str, str] = {}  # "session_id:character_name" -> system_prompt

def get_retriever(session_id: str) -> Optional[any]:
    """세션의 retriever 가져오기"""
    return vector_store_mapping.get(session_id)

def set_retriever(session_id: str, retriever: any):
    """세션의 retriever 저장"""
    vector_store_mapping[session_id] = retriever

def get_system_prompt(session_id: str, character_name: str) -> Optional[str]:
    """세션+캐릭터의 시스템 프롬프트 가져오기"""
    key = f"{session_id}:{character_name}"
    return system_prompts.get(key)

def set_system_prompt(session_id: str, character_name: str, prompt: str):
    """세션+캐릭터의 시스템 프롬프트 저장"""
    key = f"{session_id}:{character_name}"
    system_prompts[key] = prompt
