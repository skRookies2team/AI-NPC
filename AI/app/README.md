# AI NPC Server - Refactored Module Structure

ai_server.py가 여러 모듈로 리팩토링되었습니다. 기존의 모든 기능을 유지하면서 코드 구조가 개선되었습니다.

## 디렉토리 구조

```
app/
├── __init__.py                 # 패키지 초기화
├── main.py                     # FastAPI 앱 팩토리
├── config/                     # 설정
│   ├── __init__.py
│   └── settings.py             # 환경변수 및 설정값
├── core/                       # 핵심 유틸리티
│   ├── __init__.py
│   ├── logging.py              # 로깅 설정
│   └── state.py                # 전역 상태 관리
├── database/                   # 데이터베이스
│   ├── __init__.py
│   ├── connection.py           # DB 연결
│   └── session.py              # 세션 CRUD
├── middleware/                 # 미들웨어
│   ├── __init__.py
│   └── logging.py              # 요청/응답 로깅
├── models/                     # 데이터 모델
│   ├── __init__.py
│   └── requests.py             # Pydantic 요청 모델
├── services/                   # 비즈니스 로직
│   ├── __init__.py
│   ├── training.py             # 학습 서비스
│   ├── character.py            # 캐릭터 관리
│   ├── chat.py                 # 채팅/RAG 실행
│   └── vectorstore.py          # 벡터 스토어 관리
└── api/                        # API 레이어
    ├── __init__.py
    ├── routes.py               # 라우트 등록
    └── endpoints/              # 엔드포인트 핸들러
        ├── __init__.py
        ├── training.py         # /train-from-s3, /train-text
        ├── character.py        # /character
        ├── chat.py             # /chat
        ├── update.py           # /update
        └── sessions.py         # /sessions, /session/{id}
```

## 모듈별 설명

### config/settings.py
- 환경변수 로드 및 검증
- OpenAI API 키, PostgreSQL 연결 문자열, AWS 자격 증명
- RAG 설정 (chunk_size, temperature 등)
- 서버 설정 (host, port)

### core/logging.py
- 로깅 설정 및 logger 인스턴스
- 파일 및 콘솔 핸들러
- 시작 로그 출력

### core/state.py
- 인메모리 전역 상태 관리
- `vector_store_mapping`: session_id → retriever
- `system_prompts`: "session_id:character_name" → 프롬프트
- 접근 함수: `get_retriever`, `set_retriever`, `get_system_prompt`, `set_system_prompt`

### database/connection.py
- PostgreSQL 연결 함수
- 연결 문자열 파싱

### database/session.py
- 세션 정보 테이블 관리
- CRUD 함수: `init_session_table`, `save_session_info`, `load_session_info`, `load_all_characters_for_session`, `list_all_sessions`

### middleware/logging.py
- 요청/응답 로깅 미들웨어
- 처리 시간 측정
- 에러 로깅

### models/requests.py
- Pydantic 요청 모델
- `ChatRequest`, `UpdateContentRequest`, `TrainFromS3Request`, `CharacterRequest`, `TrainTextRequest`

### services/vectorstore.py
- 벡터 스토어 생성, 로드, 문서 추가
- `create_vectorstore`, `load_vectorstore`, `add_documents_to_vectorstore`

### services/training.py
- S3 다운로드, 텍스트 로드 및 청킹
- `download_from_s3`, `load_and_chunk_text`, `save_text_to_temp_file`, `cleanup_temp_file`

### services/character.py
- 캐릭터 관리 및 프롬프트 생성
- `generate_system_prompt`, `set_character_info`

### services/chat.py
- RAG 체인 실행
- 레거시 session_id 형식 지원
- `execute_chat`

### api/endpoints/*
- 각 엔드포인트 핸들러
- 서비스 레이어 호출
- 에러 처리 및 로깅

### api/routes.py
- 모든 라우트를 FastAPI 앱에 등록

### main.py
- FastAPI 앱 팩토리
- 미들웨어, 에러 핸들러, 라우트 등록
- 시작 이벤트 (테이블 초기화)

## 의존성 흐름

```
config.settings (최하위)
    ↓
core.logging, core.state
    ↓
database.connection
    ↓
database.session
    ↓
models.requests, middleware.logging
    ↓
services.*
    ↓
api.endpoints.*
    ↓
api.routes
    ↓
main
```

순환 의존성이 없는 깔끔한 단방향 흐름입니다.

## 사용법

### 서버 시작
```bash
python ai_server.py
```

### 모듈 import
```python
from app.config import settings
from app.core.logging import logger
from app.services.chat import execute_chat
from app.database.session import save_session_info
```

## 주요 변경사항

### 이전 (ai_server.py - 1043줄)
- 모든 코드가 하나의 파일에 집중
- 설정, 데이터베이스, 비즈니스 로직, API 모두 혼재
- 테스트 및 유지보수 어려움

### 현재 (리팩토링 후)
- **20개의 모듈로 분리** (각 50-100줄)
- 명확한 관심사 분리
- 테스트 가능한 서비스 레이어
- 재사용 가능한 컴포넌트

## 하위 호환성

모든 기능이 100% 유지됩니다:
- ✅ 모든 API 엔드포인트 동일
- ✅ 데이터베이스 스키마 변경 없음
- ✅ 환경변수 (.env) 그대로 사용
- ✅ 로그 형식 동일
- ✅ 전역 상태 (벡터 스토어, 프롬프트) 동일하게 동작

## 향후 개선 가능 사항

리팩토링된 구조로 인해 다음이 쉬워집니다:
- 유닛 테스트 추가
- Redis 캐싱 (core/state.py만 수정)
- API 버저닝 (api/v1, api/v2)
- 의존성 주입 (DI)
- OpenTelemetry 추적
- 비동기 데이터베이스 (asyncpg)

## 롤백

문제 발생 시:
```bash
# 백업 파일로 복원
cp ai_server.py.backup ai_server.py
# 서버 재시작
python ai_server.py
```

## 참고

- 원본 파일: `ai_server.py.backup`
- 리팩토링 계획: `.claude/plans/whimsical-watching-chipmunk.md`
