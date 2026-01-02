from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from app.config import settings
from app.core.state import get_retriever, get_system_prompt
from app.core.logging import logger
from app.services.vectorstore import load_vectorstore
from app.database.session import load_session_info
import traceback

def execute_chat(session_id: str, character_name: str, message: str) -> str:
    """RAG 체인을 실행하여 채팅 메시지 처리

    Raises:
        ValueError: 세션 또는 캐릭터를 찾을 수 없는 경우
    """
    # session_id 파싱 (레거시 형식 처리: 캐릭터 이름이 포함된 경우)
    base_session_id = session_id
    extracted_character = None

    if '_' in session_id and session_id.startswith('story_') and not character_name:
        parts = session_id.split('_')
        if len(parts) >= 3:
            base_session_id = '_'.join(parts[:2])
            extracted_character = '_'.join(parts[2:])
            logger.debug(f"session_id에서 캐릭터 추출: {extracted_character}")

    if extracted_character:
        character_name = extracted_character
        session_id = base_session_id

    # Retriever 로드 (메모리 또는 PostgreSQL에서)
    retriever = get_retriever(session_id)
    if not retriever:
        # 데이터베이스에서 로드 시도
        vectorstore = load_vectorstore(session_id)
        if not vectorstore:
            # 원본 session_id로도 시도 (파싱 실패 케이스)
            if base_session_id != session_id:
                vectorstore = load_vectorstore(base_session_id)
                if vectorstore:
                    session_id = base_session_id

        if not vectorstore:
            raise ValueError(f"세션을 찾을 수 없습니다: {session_id}")

        retriever = get_retriever(session_id)

    # 시스템 프롬프트 로드
    template = get_system_prompt(session_id, character_name)
    if not template:
        session_info = load_session_info(session_id, character_name)
        if session_info:
            template = session_info['system_prompt']
            logger.info(f"PostgreSQL에서 시스템 프롬프트 로드 완료 - session_id: {session_id}, character_name: {character_name}")
        else:
            # 폴백 템플릿
            logger.warning(f"세션 정보를 찾을 수 없습니다 - session_id: {session_id}, character_name: {character_name}")
            template = f"""당신은 소설 속 인물 '{character_name}'입니다. 당신은 이제 게임 속 NPC로서 플레이어와 직접 대화하고 있습니다.

**대화 규칙:**
1. 캐릭터로서 직접 말하세요. 3인칭 설명 절대 금지.
2. 설명하는 AI가 아니라 살아있는 캐릭터처럼 자연스럽게 대화하세요.
3. "{character_name}는..."이 아니라 "나는..."이라고 말하세요.

[Context]:
{{context}}"""

    # 프롬프트 구성
    prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("human", "{question}")
    ])

    # LLM 생성
    llm = ChatOpenAI(
        model=settings.MODEL_NAME,
        temperature=settings.TEMPERATURE,
        openai_api_key=settings.OPENAI_API_KEY
    )

    # 문서 포맷팅 함수
    def format_docs(docs):
        formatted = "\n\n".join(doc.page_content for doc in docs)
        logger.debug(f"검색된 문서 수: {len(docs)}, 컨텍스트 길이: {len(formatted)} 문자")
        if docs:
            logger.debug(f"첫 번째 문서 미리보기: {docs[0].page_content[:200]}...")
        return formatted

    # 검색 테스트
    logger.debug("RAG 체인 실행 중...")
    logger.info(f"[CHAT] 사용할 시스템 프롬프트 (처음 300자): {template[:300]}...")
    logger.info(f"[CHAT] 프롬프트에 context 변수가 있는지: {'{context}' in template}")

    try:
        test_docs = retriever.invoke(message)
        logger.info(f"[CHAT] 벡터 검색 성공 - 검색된 문서 수: {len(test_docs)}")
        if len(test_docs) == 0:
            logger.warning("⚠️ [CHAT] 벡터 검색 결과가 비어있습니다! 소설 학습이 제대로 되지 않았을 수 있습니다.")
        else:
            logger.info(f"[CHAT] 첫 번째 검색 결과 미리보기: {test_docs[0].page_content[:200]}...")
    except Exception as e:
        logger.error(f"[CHAT] 벡터 검색 테스트 실패: {str(e)}")
        logger.error(f"[CHAT] Traceback: {traceback.format_exc()}")

    # RAG 체인 구성 및 실행
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = rag_chain.invoke(message)
    logger.debug(f"생성된 응답 길이: {len(response)} 문자")

    return response
