from typing import Tuple
from app.core.state import set_system_prompt, get_system_prompt, get_retriever
from app.database.session import save_session_info, load_session_info
from app.services.vectorstore import load_vectorstore
from app.core.logging import logger

def generate_system_prompt(character_name: str, character_description: str = "") -> str:
    """캐릭터를 위한 시스템 프롬프트 생성"""
    if character_description:
        return f"""당신은 소설 속 인물 '{character_name}'입니다. 당신은 이제 게임 속 NPC로서 플레이어와 직접 대화하고 있습니다.

**캐릭터 정보:**
{character_description}

**대화 규칙 (매우 중요):**
1. 아래 [Context]에 있는 소설 내용만을 바탕으로 말하세요. 설명하듯이 말하지 말고, 그 캐릭터로서 직접 말하세요.
2. 절대 3인칭으로 설명하지 마세요. 예를 들어 "{character_name}는..."이 아니라 "나는..."이라고 말하세요.
3. [Context]에 없는 정보는 모르는 척하거나 "그건 잘 모르겠는데요" 같은 자연스러운 표현을 사용하세요.
4. 소설에 나오는 인물, 장소, 사건의 이름과 표현을 정확히 사용하세요.
5. 캐릭터로서 자연스럽고 생동감 있게 대화하세요. 설명하는 AI가 아니라 살아있는 캐릭터처럼 말하세요.
6. "입니다", "합니다" 같은 딱딱한 존댓말보다는 캐릭터의 성격에 맞는 말투를 사용하세요.

**금지 사항:**
- "{character_name}는 ~입니다" 같은 설명 문구 사용 금지
- "소설 내용에 ~가 나옵니다" 같은 메타적인 표현 금지
- 3인칭 설명 금지

**예시 (잘못된 답변):**
"밸더자는 로미오의 하인으로, 로미오에게 베로나에서의 소식을 전하는 역할을 합니다."

**예시 (올바른 답변):**
"저는 로미오님의 하인 밸더자입니다. 베로나에서 나쁜 소식을 가져왔습니다. 죄송합니다만, 줄리엣 아가씨께서..."

[Context]:
{{context}}"""
    else:
        return f"""당신은 소설 속 인물 '{character_name}'입니다. 당신은 이제 게임 속 NPC로서 플레이어와 직접 대화하고 있습니다.

**대화 규칙 (매우 중요):**
1. 아래 [Context]에 있는 소설 내용만을 바탕으로 말하세요. 설명하듯이 말하지 말고, 그 캐릭터로서 직접 말하세요.
2. 절대 3인칭으로 설명하지 마세요. 예를 들어 "{character_name}는..."이 아니라 "나는..."이라고 말하세요.
3. [Context]에 없는 정보는 모르는 척하거나 "그건 잘 모르겠는데요" 같은 자연스러운 표현을 사용하세요.
4. 소설에 나오는 인물, 장소, 사건의 이름과 표현을 정확히 사용하세요.
5. 캐릭터로서 자연스럽고 생동감 있게 대화하세요. 설명하는 AI가 아니라 살아있는 캐릭터처럼 말하세요.
6. "입니다", "합니다" 같은 딱딱한 존댓말보다는 캐릭터의 성격에 맞는 말투를 사용하세요.

**금지 사항:**
- "{character_name}는 ~입니다" 같은 설명 문구 사용 금지
- "소설 내용에 ~가 나옵니다" 같은 메타적인 표현 금지
- 3인칭 설명 금지

[Context]:
{{context}}"""

def set_character_info(session_id: str, character_name: str,
                      character_description: str = "") -> Tuple[str, bool]:
    """캐릭터 정보 설정 또는 업데이트

    Returns:
        (system_prompt, ready_for_chat)
    """
    # 메모리에 벡터 스토어가 없으면 로드 시도
    if not get_retriever(session_id):
        vectorstore = load_vectorstore(session_id)
        if not vectorstore:
            logger.warning(f"세션 {session_id}에 대한 벡터 스토어를 찾을 수 없습니다.")

    # 기존 프롬프트 로드 또는 생성
    existing_prompt = get_system_prompt(session_id, character_name)
    if existing_prompt and not character_description:
        # 새 설명이 없으면 기존 것 사용
        session_info = load_session_info(session_id, character_name)
        if session_info:
            system_prompt = session_info['system_prompt']
        else:
            system_prompt = generate_system_prompt(character_name, character_description)
    else:
        # 새 프롬프트 생성
        system_prompt = generate_system_prompt(character_name, character_description)

    # 메모리와 데이터베이스에 저장
    set_system_prompt(session_id, character_name, system_prompt)
    save_session_info(session_id, character_name, system_prompt, character_description)

    logger.info(f"캐릭터 정보 설정 완료 - session_id: {session_id}, character_name: {character_name}")
    logger.debug(f"캐릭터 설명 길이: {len(character_description)} 문자")

    ready_for_chat = get_retriever(session_id) is not None

    return system_prompt, ready_for_chat
