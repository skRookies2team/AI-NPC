import os
import shutil
import logging
import traceback
import re
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

# LangChain 관련 임포트
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# 1. 환경변수 로드 (.env 파일에서 API 키 가져옴)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
POSTGRES_CONNECTION_STRING = os.getenv("POSTGRES_CONNECTION_STRING")

# AWS 자격 증명 (환경변수에서 가져오기)
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_S3_REGION", "ap-northeast-2")
S3_BUCKET = os.getenv("AWS_S3_BUCKET")  # S3 버킷 이름 (환경변수에서만 가져옴)

if not S3_BUCKET:
    raise ValueError(".env 파일에 AWS_S3_BUCKET 또는 S3_BUCKET이 설정되지 않았습니다.")

if not OPENAI_API_KEY:
    raise ValueError(".env 파일에 OPENAI_API_KEY가 설정되지 않았습니다.")

if not POSTGRES_CONNECTION_STRING:
    raise ValueError(".env 파일에 POSTGRES_CONNECTION_STRING이 설정되지 않았습니다.")

# 2. 로깅 설정
os.makedirs("logs", exist_ok=True)
log_file = os.path.join("logs", "ai_server.log")

# 로거 설정
logger = logging.getLogger("ai_server")
logger.setLevel(logging.INFO)

# 파일 핸들러 (로그 파일에 저장)
file_handler = logging.FileHandler(log_file, encoding="utf-8")
file_handler.setLevel(logging.INFO)

# 콘솔 핸들러 (터미널에 출력)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 포맷터 설정
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 핸들러 추가
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info("=" * 50)
logger.info("AI 서버 시작")
logger.info("=" * 50)
logger.info(f"S3_BUCKET: {S3_BUCKET}")
logger.info(f"AWS_REGION: {AWS_REGION}")
logger.info(f"AWS_ACCESS_KEY 설정됨: {bool(AWS_ACCESS_KEY_ID)}")
logger.info(f"AWS_SECRET_KEY 설정됨: {bool(AWS_SECRET_ACCESS_KEY)}")

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

# PostgreSQL 세션 정보 저장/로드 함수
def get_db_connection():
    """PostgreSQL 연결 문자열에서 연결 객체 생성"""
    # connection_string 형식: "postgresql://user:password@host:port/database"
    # psycopg2는 "postgresql://" 대신 키워드 인자나 직접 파싱 필요
    # URL 파싱
    from urllib.parse import urlparse
    parsed = urlparse(POSTGRES_CONNECTION_STRING)
    return psycopg2.connect(
        host=parsed.hostname,
        port=parsed.port or 5432,
        database=parsed.path[1:] if parsed.path else None,  # '/' 제거
        user=parsed.username,
        password=parsed.password
    )

def init_session_table():
    """세션 정보 저장 테이블 생성 (한 세션에 여러 캐릭터 지원)"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # 기존 테이블이 있는지 확인
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'session_info'
            );
        """)
        table_exists = cur.fetchone()[0]

        if table_exists:
            # 기존 테이블이 있으면 PRIMARY KEY 변경 시도
            try:
                # 기존 PRIMARY KEY 제약조건 확인 및 제거
                cur.execute("""
                    SELECT constraint_name
                    FROM information_schema.table_constraints
                    WHERE table_name = 'session_info'
                    AND constraint_type = 'PRIMARY KEY';
                """)
                pk_constraint = cur.fetchone()
                if pk_constraint:
                    cur.execute(f"ALTER TABLE session_info DROP CONSTRAINT {pk_constraint[0]};")
                    logger.info(f"기존 PRIMARY KEY 제약조건 제거: {pk_constraint[0]}")

                # 복합 PRIMARY KEY 추가
                cur.execute("""
                    ALTER TABLE session_info
                    ADD PRIMARY KEY (session_id, character_name);
                """)
                logger.info("기존 테이블에 복합 PRIMARY KEY 추가 완료")
            except Exception as e:
                # 이미 복합 키가 있거나 다른 이유로 실패할 수 있음
                logger.debug(f"PRIMARY KEY 변경 시도 중 (이미 변경되었을 수 있음): {str(e)}")
                conn.rollback()
        else:
            # 새 테이블 생성
            cur.execute("""
                CREATE TABLE session_info (
                    session_id VARCHAR(255) NOT NULL,
                    character_name VARCHAR(255) NOT NULL,
                    system_prompt TEXT NOT NULL,
                    character_description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (session_id, character_name)
                )
            """)
            logger.info("새 세션 정보 테이블 생성 완료 (복합 PRIMARY KEY)")

        conn.commit()
        cur.close()
        conn.close()
        logger.info("세션 정보 테이블 초기화 완료")
    except Exception as e:
        logger.error(f"세션 정보 테이블 초기화 실패: {str(e)}")
        raise

def save_session_info(session_id: str, character_name: str, system_prompt: str, character_description: str = None):
    """세션 정보를 PostgreSQL에 저장 (한 세션에 여러 캐릭터 지원)"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO session_info (session_id, character_name, system_prompt, character_description, updated_at)
            VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (session_id, character_name)
            DO UPDATE SET
                system_prompt = EXCLUDED.system_prompt,
                character_description = EXCLUDED.character_description,
                updated_at = CURRENT_TIMESTAMP
        """, (session_id, character_name, system_prompt, character_description))
        conn.commit()
        cur.close()
        conn.close()
        logger.debug(f"세션 정보 저장 완료 - session_id: {session_id}, character_name: {character_name}")
    except Exception as e:
        logger.error(f"세션 정보 저장 실패 - session_id: {session_id}, character_name: {character_name}, error: {str(e)}")
        raise

def load_session_info(session_id: str, character_name: str = None):
    """PostgreSQL에서 세션 정보 로드 (특정 캐릭터 또는 세션의 모든 캐릭터)"""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        if character_name:
            # 특정 캐릭터만 로드
            cur.execute("""
                SELECT session_id, character_name, system_prompt, character_description, created_at, updated_at
                FROM session_info
                WHERE session_id = %s AND character_name = %s
            """, (session_id, character_name))
            result = cur.fetchone()
            cur.close()
            conn.close()
            if result:
                logger.debug(f"세션 정보 로드 완료 - session_id: {session_id}, character_name: {character_name}")
                return dict(result)
            return None
        else:
            # 세션의 모든 캐릭터 로드 (첫 번째 캐릭터 반환 - 하위 호환성)
            cur.execute("""
                SELECT session_id, character_name, system_prompt, character_description, created_at, updated_at
                FROM session_info
                WHERE session_id = %s
                ORDER BY updated_at DESC
                LIMIT 1
            """, (session_id,))
            result = cur.fetchone()
            cur.close()
            conn.close()
            if result:
                logger.debug(f"세션 정보 로드 완료 - session_id: {session_id} (첫 번째 캐릭터)")
                return dict(result)
            return None
    except Exception as e:
        logger.warning(f"세션 정보 로드 실패 - session_id: {session_id}, character_name: {character_name}, error: {str(e)}")
        return None

def load_all_characters_for_session(session_id: str):
    """세션의 모든 캐릭터 정보 로드"""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT session_id, character_name, system_prompt, character_description, created_at, updated_at
            FROM session_info
            WHERE session_id = %s
            ORDER BY updated_at DESC
        """, (session_id,))
        results = cur.fetchall()
        cur.close()
        conn.close()
        characters = [dict(row) for row in results]
        logger.debug(f"세션의 모든 캐릭터 로드 완료 - session_id: {session_id}, 캐릭터 수: {len(characters)}")
        return characters
    except Exception as e:
        logger.warning(f"세션의 모든 캐릭터 로드 실패 - session_id: {session_id}, error: {str(e)}")
        return []

def list_all_sessions():
    """PostgreSQL에서 모든 세션 목록 조회"""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT session_id, character_name, character_description, created_at, updated_at
            FROM session_info
            ORDER BY updated_at DESC
        """)
        results = cur.fetchall()
        cur.close()
        conn.close()
        sessions = [dict(row) for row in results]
        logger.debug(f"세션 목록 조회 완료 - 총 {len(sessions)}개 세션")
        return sessions
    except Exception as e:
        logger.error(f"세션 목록 조회 실패 - error: {str(e)}")
        return []

# 서버 시작 시 테이블 초기화
init_session_table()

# 요청/응답 로깅 미들웨어
class LoggingMiddleware(BaseHTTPMiddleware):
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

app.add_middleware(LoggingMiddleware)

# 간단한 인메모리 세션 저장소
vector_store_mapping = {}
system_prompts = {}

class ChatRequest(BaseModel):
    session_id: str
    character_name: str  # 대화할 NPC 캐릭터 이름 (필수)
    message: str

class UpdateContentRequest(BaseModel):
    session_id: str
    content: str  # 추가할 소설 내용 (텍스트)
    metadata: dict = {}  # 선택적 메타데이터 (예: 챕터 번호, 타임스탬프 등)

class TrainFromS3Request(BaseModel):
    session_id: str
    file_key: str  # S3 파일 키
    bucket: str  # S3 버킷 이름
    character_name: str

class CharacterRequest(BaseModel):
    session_id: str
    character_name: str
    character_description: str = ""  # 선택적: 캐릭터의 성격, 특징 등 추가 정보

@app.post("/api/ai/train-from-s3")
async def train_novel_from_s3(request: TrainFromS3Request):
    """
    S3에서 파일을 다운로드하여 학습
    프론트엔드가 S3에 업로드한 파일을 다운로드하여 RAG 학습 진행
    """
    # 환경변수에서 버킷 이름 가져오기
    bucket_name = S3_BUCKET

    logger.info(f"[TRAIN-FROM-S3] 시작 - session_id: {request.session_id}, file_key: {request.file_key}, bucket: {bucket_name}")
    file_path = None

    try:
        # boto3 임포트 확인
        try:
            import boto3
            from botocore.exceptions import ClientError, NoCredentialsError
        except ImportError:
            logger.error("boto3가 설치되지 않았습니다. pip install boto3를 실행해주세요.")
            raise HTTPException(status_code=500, detail="boto3가 설치되지 않았습니다.")

        # S3 클라이언트 생성 (환경변수에서 자격 증명 읽기)
        if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
            logger.error("AWS 자격 증명이 .env 파일에 설정되지 않았습니다.")
            logger.error(f"AWS_ACCESS_KEY_ID: {bool(AWS_ACCESS_KEY_ID)}, AWS_SECRET_ACCESS_KEY: {bool(AWS_SECRET_ACCESS_KEY)}")
            raise HTTPException(
                status_code=500,
                detail="AWS 자격 증명이 설정되지 않았습니다. .env 파일에 AWS_ACCESS_KEY_ID와 AWS_SECRET_ACCESS_KEY를 추가하세요."
            )

        try:
            s3_client = boto3.client(
                's3',
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                region_name=AWS_REGION
            )
            logger.debug("환경변수에서 AWS 자격 증명 사용")
        except Exception as e:
            logger.error(f"S3 클라이언트 생성 실패: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"AWS S3 클라이언트 생성 실패: {str(e)}"
            )

        # 임시 파일 경로 생성
        os.makedirs("temp_ai", exist_ok=True)
        filename = os.path.basename(request.file_key) or "novel.txt"
        file_path = f"temp_ai/{request.session_id}_{filename}"

        # S3에서 파일 다운로드
        logger.debug(f"S3에서 파일 다운로드 중 - bucket: {bucket_name}, key: {request.file_key}")
        try:
            s3_client.download_file(bucket_name, request.file_key, file_path)
            logger.info(f"파일 다운로드 완료 - {file_path}")
        except ClientError as e:
            logger.error(f"S3 파일 다운로드 실패 - error: {str(e)}")
            raise HTTPException(
                status_code=404,
                detail=f"S3에서 파일을 찾을 수 없습니다. bucket: {bucket_name}, key: {request.file_key}"
            )

        # 텍스트 로드
        logger.debug("텍스트 파일 로드 중...")
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        logger.info(f"로드된 문서 수: {len(docs)}, 총 문자 수: {sum(len(doc.page_content) for doc in docs)}")

        # 텍스트 청킹
        logger.debug("텍스트 청킹 중...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        splits = text_splitter.split_documents(docs)
        logger.info(f"청킹 완료 - 총 {len(splits)}개 청크 생성")

        # 임베딩 및 벡터 저장
        logger.debug("임베딩 생성 및 벡터 저장소 생성 중...")
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        # PostgreSQL Vector Store 사용
        collection_name = f"session_{request.session_id}"
        vectorstore = PGVector.from_documents(
            documents=splits,
            embedding=embeddings,
            collection_name=collection_name,
            connection_string=POSTGRES_CONNECTION_STRING,
            use_jsonb=True
        )
        logger.info(f"벡터 저장소 생성 완료 - collection: {collection_name}")

        # Retriever 설정
        vector_store_mapping[request.session_id] = vectorstore.as_retriever(
            search_kwargs={"k": 6}
        )

        # train-from-s3에서는 소설만 학습하고, 캐릭터 이름은 나중에 /api/ai/character에서 설정
        # 여기서는 기본 프롬프트만 생성 (캐릭터 이름 없이)
        # character_name은 나중에 설정될 때까지 기본값 사용
        character_name = request.character_name.strip() if request.character_name else "캐릭터"

        # 기본 프롬프트 설정 (캐릭터 이름은 나중에 /api/ai/character에서 설정됨)
        system_prompt = f"""
        당신은 소설 속 인물입니다.

        **중요 규칙:**
        1. 아래 [Context]에 있는 소설 내용만을 바탕으로 답변하세요.
        2. [Context]에 없는 정보는 절대 만들어내지 마세요.
        3. [Context]에 답변할 수 있는 정보가 없으면 "소설 내용에 그런 정보는 나오지 않습니다" 또는 "모르겠습니다"라고 솔직하게 말하세요.
        4. 컨텍스트 밖의 일반 지식이나 추측을 사용하지 마세요.
        5. 소설에 나오는 인물, 장소, 사건의 이름과 표현을 정확히 사용하세요.

        [Context]:
        {{context}}
        """

        # train-from-s3에서는 임시 프롬프트만 저장 (나중에 /api/ai/character에서 제대로 설정됨)
        # 새로운 키 형식 사용 (하지만 character_name이 임시 값이므로 /api/ai/character에서 덮어쓰게 됨)
        prompt_key = f"{request.session_id}:{character_name}"
        system_prompts[prompt_key] = system_prompt

        # PostgreSQL에 세션 정보 저장 (character_name은 임시로 저장, 나중에 /api/ai/character에서 업데이트됨)
        save_session_info(
            session_id=request.session_id,
            character_name=character_name,  # 임시 값 (나중에 /api/ai/character에서 업데이트)
            system_prompt=system_prompt,
            character_description=None
        )
        logger.info(f"[TRAIN-FROM-S3] 소설 학습 완료 - session_id: {request.session_id}")
        logger.info(f"[TRAIN-FROM-S3] 캐릭터 이름은 /api/ai/character 엔드포인트에서 설정해주세요.")

        # 임시 파일 삭제
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"임시 파일 삭제 완료 - {file_path}")

        logger.info(f"[TRAIN-FROM-S3] 완료 - session_id: {request.session_id}")
        return {
            "status": "trained",
            "session_id": request.session_id,
            "chunks_created": len(splits)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[TRAIN-FROM-S3] 오류 발생 - session_id: {request.session_id}, error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"학습 중 오류 발생: {str(e)}")

class TrainTextRequest(BaseModel):
    """테스트용: 텍스트 직접 전송하여 학습"""
    session_id: str
    content: str  # 소설 텍스트 내용
    character_name: str = ""  # 캐릭터 이름 (선택적, 나중에 /api/ai/character에서 설정 가능)

@app.post("/api/ai/train-text")
async def train_novel_from_text(request: TrainTextRequest):
    """
    테스트용: 텍스트를 직접 받아서 학습
    실제 운영에서는 /api/ai/train-from-s3를 사용하세요
    """
    logger.info(f"[TRAIN-TEXT] 시작 - session_id: {request.session_id}")
    file_path = None

    try:
        # 임시 파일로 저장
        os.makedirs("temp_ai", exist_ok=True)
        file_path = f"temp_ai/{request.session_id}_novel.txt"

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(request.content)
        logger.info(f"임시 파일 저장 완료 - {file_path}")

        # TextLoader로 로드
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        logger.info(f"로드된 문서 수: {len(docs)}, 총 문자 수: {sum(len(doc.page_content) for doc in docs)}")

        # 텍스트 청킹
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        splits = text_splitter.split_documents(docs)
        logger.info(f"청킹 완료 - 총 {len(splits)}개 청크 생성")

        # 임베딩 및 벡터 저장
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        collection_name = f"session_{request.session_id}"
        vectorstore = PGVector.from_documents(
            documents=splits,
            embedding=embeddings,
            collection_name=collection_name,
            connection_string=POSTGRES_CONNECTION_STRING,
            use_jsonb=True
        )
        logger.info(f"벡터 저장소 생성 완료 - collection: {collection_name}")

        # Retriever 설정
        vector_store_mapping[request.session_id] = vectorstore.as_retriever(
            search_kwargs={"k": 6}
        )

        # 캐릭터 이름 처리
        character_name = request.character_name.strip() if request.character_name else "캐릭터"
        system_prompt = f"""
        당신은 소설 속 인물입니다.

        **중요 규칙:**
        1. 아래 [Context]에 있는 소설 내용만을 바탕으로 답변하세요.
        2. [Context]에 없는 정보는 절대 만들어내지 마세요.
        3. [Context]에 답변할 수 있는 정보가 없으면 "소설 내용에 그런 정보는 나오지 않습니다" 또는 "모르겠습니다"라고 솔직하게 말하세요.
        4. 컨텍스트 밖의 일반 지식이나 추측을 사용하지 마세요.
        5. 소설에 나오는 인물, 장소, 사건의 이름과 표현을 정확히 사용하세요.

        [Context]:
        {{context}}
        """

        # train-text는 소설만 학습 (character_name은 나중에 /api/ai/character에서 설정)
        # 여기서는 벡터 스토어만 생성하고, 프롬프트는 나중에 /api/ai/character에서 설정됨
        # character_name이 제공되어도 저장하지 않음 (나중에 character 엔드포인트에서 제대로 설정)
        logger.info(f"[TRAIN-TEXT] 소설 학습 완료 - session_id: {request.session_id}")
        logger.info(f"[TRAIN-TEXT] 캐릭터는 /api/ai/character 엔드포인트에서 설정하세요.")

        # 임시 파일 삭제
        if os.path.exists(file_path):
            os.remove(file_path)

        return {
            "status": "trained",
            "session_id": request.session_id,
            "chunks_created": len(splits)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[TRAIN-TEXT] 오류 발생 - session_id: {request.session_id}, error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"학습 중 오류 발생: {str(e)}")

@app.post("/api/ai/character")
async def set_character(request: CharacterRequest):
    """
    캐릭터 정보를 입력받아 시스템 프롬프트 설정
    소설 학습과 별도로 캐릭터 정보만 업데이트할 수 있음
    PostgreSQL에 벡터 스토어가 이미 존재하면 로드하여 대화 준비 완료
    """
    logger.info(f"[CHARACTER] 시작 - session_id: {request.session_id}, character_name: {request.character_name}")

    try:
        session_id = request.session_id
        character_name = request.character_name
        character_description = request.character_description.strip() if request.character_description else ""

        # 기존 벡터 스토어가 PostgreSQL에 있는지 확인하고 로드
        # (서버 재시작 등으로 메모리에 없을 수 있음)
        if session_id not in vector_store_mapping:
            logger.debug(f"메모리에 벡터 스토어 없음, PostgreSQL에서 로드 시도 - session_id: {session_id}")
            try:
                embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
                collection_name = f"session_{session_id}"
                vectorstore = PGVector(
                    collection_name=collection_name,
                    connection_string=POSTGRES_CONNECTION_STRING,
                    embedding_function=embeddings,
                    use_jsonb=True
                )
                vector_store_mapping[session_id] = vectorstore.as_retriever(
                    search_kwargs={"k": 6}
                )
                logger.info(f"PostgreSQL에서 벡터 스토어 로드 완료 - collection: {collection_name}")
            except Exception as e:
                logger.warning(f"PostgreSQL에서 벡터 스토어 로드 실패 - session_id: {session_id}, error: {str(e)}")
                logger.warning("소설 학습(train-from-s3)이 먼저 필요합니다.")
                # 벡터 스토어가 없어도 캐릭터 정보는 저장하도록 함
                # (나중에 소설 학습하면 대화 가능)

        # 기존 시스템 프롬프트 로드 (메모리에 없으면 PostgreSQL에서 로드)
        # 캐릭터 설명만 업데이트하는 경우 기존 프롬프트를 사용
        prompt_key = f"{session_id}:{character_name}"
        if prompt_key not in system_prompts:
            session_info = load_session_info(session_id, character_name)
            if session_info and not character_description:
                # 기존 프롬프트가 있고 새로운 설명이 없으면 기존 것 사용
                system_prompts[prompt_key] = session_info['system_prompt']
                logger.debug(f"기존 시스템 프롬프트 로드 완료 - session_id: {session_id}, character_name: {character_name}")

        # 캐릭터 설명이 있으면 추가, 없으면 기본 형식만 사용
        if character_description:
            system_prompt = f"""당신은 소설 속 인물 '{character_name}'입니다. 당신은 이제 게임 속 NPC로서 플레이어와 직접 대화하고 있습니다.

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
            system_prompt = f"""당신은 소설 속 인물 '{character_name}'입니다. 당신은 이제 게임 속 NPC로서 플레이어와 직접 대화하고 있습니다.

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

        # 시스템 프롬프트 저장 (메모리 + PostgreSQL)
        # 키를 session_id:character_name 형태로 저장 (한 세션에 여러 캐릭터 지원)
        prompt_key = f"{session_id}:{character_name}"
        system_prompts[prompt_key] = system_prompt

        # PostgreSQL에 세션 정보 저장
        save_session_info(
            session_id=session_id,
            character_name=character_name,
            system_prompt=system_prompt,
            character_description=character_description
        )

        logger.info(f"[CHARACTER] 완료 - session_id: {session_id}, character_name: {character_name}")
        logger.debug(f"캐릭터 설명 길이: {len(character_description)} 문자")

        # 벡터 스토어가 준비되어 있는지 확인
        ready_for_chat = session_id in vector_store_mapping

        return {
            "status": "character_set",
            "session_id": session_id,
            "character_name": character_name,
            "ready_for_chat": ready_for_chat,
            "message": "캐릭터 정보가 성공적으로 설정되었습니다." +
                      (" 대화할 준비가 완료되었습니다." if ready_for_chat else
                       " 소설 학습(train-from-s3)이 완료되면 대화할 수 있습니다.")
        }

    except Exception as e:
        logger.error(f"[CHARACTER] 오류 발생 - session_id: {request.session_id}, error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"캐릭터 정보 설정 중 오류 발생: {str(e)}")

@app.get("/api/ai/sessions")
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

@app.get("/api/ai/session/{session_id}")
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

@app.post("/api/ai/chat")
async def chat(request: ChatRequest):
    logger.info(f"[CHAT] 시작 - session_id: {request.session_id}, character_name: {request.character_name}")
    logger.debug(f"사용자 메시지: {request.message[:100]}..." if len(request.message) > 100 else f"사용자 메시지: {request.message}")

    session_id = request.session_id
    character_name = request.character_name

    # session_id에서 캐릭터 이름 부분 제거 (예: story_6bfa4631_잭 -> story_6bfa4631)
    # 프론트엔드에서 캐릭터 이름을 붙여서 보내는 경우를 처리 (하위 호환성)
    if '_' in session_id and session_id.startswith('story_') and not character_name:
        parts = session_id.split('_')
        if len(parts) >= 3:
            # 원본 session_id (캐릭터 이름 제거)
            base_session_id = '_'.join(parts[:2])  # story_6bfa4631
            character_name = '_'.join(parts[2:])  # 마지막 부분을 character_name으로 추출
            logger.debug(f"session_id에서 캐릭터 이름 추출 - base_session_id: {base_session_id}, character_name: {character_name}")
            session_id = base_session_id

    # 메모리에 없으면 PostgreSQL에서 로드 시도
    # 벡터 스토어도 원본 session_id로 찾기 시도
    vectorstore_loaded = False
    base_session_id = None
    if session_id not in vector_store_mapping:
        logger.debug(f"메모리에 벡터 스토어 없음, PostgreSQL에서 로드 시도 - session_id: {session_id}")

        # 먼저 현재 session_id로 시도
        try:
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            collection_name = f"session_{session_id}"
            vectorstore = PGVector(
                collection_name=collection_name,
                connection_string=POSTGRES_CONNECTION_STRING,
                embedding_function=embeddings,
                use_jsonb=True
            )
            # 벡터 스토어가 실제로 데이터를 가지고 있는지 확인
            test_docs = vectorstore.similarity_search("test", k=1)
            if len(test_docs) > 0:
                vector_store_mapping[session_id] = vectorstore.as_retriever(
                    search_kwargs={"k": 6}
                )
                logger.info(f"PostgreSQL에서 벡터 스토어 로드 완료 - collection: {collection_name}")
                vectorstore_loaded = True
        except Exception as e:
            logger.debug(f"현재 session_id로 벡터 스토어 로드 실패: {str(e)}")

        # 실패하면 원본 session_id로 시도 (캐릭터 이름 제거)
        if not vectorstore_loaded and '_' in request.session_id and request.session_id.startswith('story_'):
            parts = request.session_id.split('_')
            if len(parts) >= 3:
                base_session_id = '_'.join(parts[:2])  # story_6bfa4631
                try:
                    logger.debug(f"원본 session_id로 벡터 스토어 로드 시도: {base_session_id}")
                    collection_name = f"session_{base_session_id}"
                    vectorstore = PGVector(
                        collection_name=collection_name,
                        connection_string=POSTGRES_CONNECTION_STRING,
                        embedding_function=embeddings,
                        use_jsonb=True
                    )
                    test_docs = vectorstore.similarity_search("test", k=1)
                    if len(test_docs) > 0:
                        vector_store_mapping[session_id] = vectorstore.as_retriever(
                            search_kwargs={"k": 6}
                        )
                        logger.info(f"원본 session_id로 벡터 스토어 로드 완료 - collection: {collection_name}")
                        vectorstore_loaded = True
                        # session_id도 원본으로 업데이트
                        session_id = base_session_id
                except Exception as e:
                    logger.debug(f"원본 session_id로도 벡터 스토어 로드 실패: {str(e)}")

        if not vectorstore_loaded:
            logger.warning(f"[CHAT] 세션 없음 - session_id: {request.session_id}, 원본 시도: {base_session_id if base_session_id else 'N/A'}")
            raise HTTPException(
                status_code=404,
                detail="Session not found or expired. 먼저 소설 학습(train-from-s3)과 캐릭터 설정(character)을 완료해주세요."
            )

    # 시스템 프롬프트 로드 (메모리에 없으면 PostgreSQL에서 로드)
    # 키를 session_id:character_name 형태로 사용
    prompt_key = f"{session_id}:{character_name}"
    if prompt_key not in system_prompts:
        session_info = load_session_info(session_id, character_name)
        if session_info:
            system_prompts[prompt_key] = session_info['system_prompt']
            logger.info(f"PostgreSQL에서 시스템 프롬프트 로드 완료 - session_id: {session_id}, character_name: {character_name}")
            logger.debug(f"로드된 프롬프트에 context 변수 있는지: {'{context}' in session_info['system_prompt']}")
        else:
            logger.warning(f"⚠️ 세션 정보를 찾을 수 없습니다 - session_id: {session_id}, character_name: {character_name}")
            logger.warning("기본 프롬프트를 사용하지만, context가 없어 스토리 내용이 반영되지 않을 수 있습니다.")

    try:
        retriever = vector_store_mapping[session_id]
        template = system_prompts.get(prompt_key, f"""당신은 소설 속 인물 '{character_name}'입니다. 당신은 이제 게임 속 NPC로서 플레이어와 직접 대화하고 있습니다.

**대화 규칙:**
1. 캐릭터로서 직접 말하세요. 3인칭 설명 절대 금지.
2. 설명하는 AI가 아니라 살아있는 캐릭터처럼 자연스럽게 대화하세요.
3. "{character_name}는..."이 아니라 "나는..."이라고 말하세요.

[Context]:
{{context}}""")

        prompt = ChatPromptTemplate.from_messages([
            ("system", template),
            ("human", "{question}")
        ])

        # 서버의 환경변수 키 사용
        # temperature를 높여서 더 자연스럽고 창의적인 NPC 대화 가능하게 설정
        llm = ChatOpenAI(model="gpt-4o", temperature=0.8, openai_api_key=OPENAI_API_KEY)

        def format_docs(docs):
            formatted = "\n\n".join(doc.page_content for doc in docs)
            logger.debug(f"검색된 문서 수: {len(docs)}, 컨텍스트 길이: {len(formatted)} 문자")
            if docs:
                logger.debug(f"첫 번째 문서 미리보기: {docs[0].page_content[:200]}...")
            return formatted

        logger.debug("RAG 체인 실행 중...")

        # 디버깅: 프롬프트 템플릿 확인
        logger.info(f"[CHAT] 사용할 시스템 프롬프트 (처음 300자): {template[:300]}...")
        logger.info(f"[CHAT] 프롬프트에 context 변수가 있는지: {'{context}' in template}")

        # 디버깅: 벡터 검색 테스트
        try:
            test_docs = retriever.invoke(request.message)
            logger.info(f"[CHAT] 벡터 검색 성공 - 검색된 문서 수: {len(test_docs)}")
            if len(test_docs) == 0:
                logger.warning("⚠️ [CHAT] 벡터 검색 결과가 비어있습니다! 소설 학습이 제대로 되지 않았을 수 있습니다.")
            else:
                logger.info(f"[CHAT] 첫 번째 검색 결과 미리보기: {test_docs[0].page_content[:200]}...")
        except Exception as e:
            logger.error(f"[CHAT] 벡터 검색 테스트 실패: {str(e)}")
            logger.error(f"[CHAT] Traceback: {traceback.format_exc()}")

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        response = rag_chain.invoke(request.message)
        logger.info(f"[CHAT] 완료 - session_id: {session_id}")
        logger.debug(f"AI 응답: {response[:100]}..." if len(response) > 100 else f"AI 응답: {response}")
        return {"reply": response}

    except Exception as e:
        logger.error(f"[CHAT] 오류 발생 - session_id: {session_id}, error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"채팅 처리 중 오류 발생: {str(e)}")

@app.post("/api/ai/update")
async def update_novel_content(request: UpdateContentRequest):
    """
    실시간으로 변화하는 소설 상황을 RAG에 추가
    게임 진행 중 발생하는 새로운 이벤트나 스토리 내용을 벡터 DB에 추가하여 학습
    """
    logger.info(f"[UPDATE] 시작 - session_id: {request.session_id}")
    logger.debug(f"추가할 내용 길이: {len(request.content)} 문자")

    try:
        session_id = request.session_id

        # 기존 벡터 스토어 로드
        logger.debug("벡터 스토어 로드 중...")
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        collection_name = f"session_{session_id}"

        # 기존 벡터 스토어가 존재하는지 확인
        try:
            vectorstore = PGVector(
                collection_name=collection_name,
                connection_string=POSTGRES_CONNECTION_STRING,
                embedding_function=embeddings,
                use_jsonb=True
            )
        except Exception as e:
            # 벡터 스토어가 없으면 새로 생성
            logger.warning(f"[UPDATE] 세션 없음 - session_id: {session_id}, error: {str(e)}")
            raise HTTPException(
                status_code=404,
                detail=f"세션이 존재하지 않습니다. 먼저 /api/ai/train으로 초기 학습을 진행해주세요. Error: {str(e)}"
            )

        # 새로 추가할 텍스트를 문서로 변환
        new_content = request.content.strip()
        if not new_content:
            logger.warning(f"[UPDATE] 빈 내용 - session_id: {session_id}")
            raise HTTPException(status_code=400, detail="추가할 내용이 비어있습니다.")

        # 텍스트를 Document 객체로 변환 (메타데이터 포함)
        new_doc = Document(
            page_content=new_content,
            metadata={
                **request.metadata,
                "added_at": str(datetime.now()),
                "source": "realtime_update"
            }
        )
        logger.debug(f"메타데이터: {new_doc.metadata}")

        # 텍스트 청킹 (기존과 동일한 설정 사용)
        logger.debug("텍스트 청킹 중...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        splits = text_splitter.split_documents([new_doc])
        logger.info(f"청킹 완료 - {len(splits)}개 청크 생성")

        # 벡터 스토어에 새 문서 추가
        logger.debug("벡터 스토어에 문서 추가 중...")
        vectorstore.add_documents(splits)

        # 메모리 캐시의 retriever도 업데이트 (존재하는 경우)
        if session_id in vector_store_mapping:
            vector_store_mapping[session_id] = vectorstore.as_retriever(
                search_kwargs={"k": 6}
            )
            logger.debug("메모리 캐시 retriever 업데이트 완료")

        logger.info(f"[UPDATE] 완료 - session_id: {session_id}, chunks_added: {len(splits)}")
        return {
            "status": "updated",
            "session_id": session_id,
            "chunks_added": len(splits),
            "message": f"새로운 소설 내용 {len(splits)}개 청크가 성공적으로 추가되었습니다."
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[UPDATE] 오류 발생 - session_id: {request.session_id}, error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"내용 추가 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    logger.info("서버 시작 - 포트: 8002")
    # AI 서버 포트: 8002
    uvicorn.run(app, host="0.0.0.0", port=8002)
