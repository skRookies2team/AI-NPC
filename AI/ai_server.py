import os
import shutil
import logging
import traceback
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

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

app = FastAPI()

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

class TrainFromS3Request(BaseModel):
    session_id: str
    file_key: str  # S3 파일 키
    bucket: str  # S3 버킷 이름
    character_name: str

@app.post("/api/ai/train")
async def train_novel(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    character_name: str = Form(...)
):
    logger.info(f"[TRAIN] 시작 - session_id: {session_id}, character_name: {character_name}, filename: {file.filename}")
    try:
        # 파일 임시 저장
        os.makedirs("temp_ai", exist_ok=True)
        file_path = f"temp_ai/{session_id}_{file.filename}"
        logger.debug(f"파일 임시 저장 중: {file_path}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 텍스트 로드
        logger.debug("텍스트 파일 로드 중...")
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        logger.info(f"로드된 문서 수: {len(docs)}, 총 문자 수: {sum(len(doc.page_content) for doc in docs)}")
        
        # 텍스트 청킹 (Chunking) - 최적화된 크기
        # chunk_size: 800 (더 작은 청크로 정확도 향상)
        # chunk_overlap: 150 (문맥 유지를 위한 적절한 오버랩)
        logger.debug("텍스트 청킹 중...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""]  # 문단 > 문장 > 단어 순서로 분할
        )
        splits = text_splitter.split_documents(docs)
        logger.info(f"청킹 완료 - 총 {len(splits)}개 청크 생성")

        # 임베딩 및 벡터 저장 (서버의 환경변수 키 사용)
        logger.debug("임베딩 생성 및 벡터 저장소 생성 중...")
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        
        # PostgreSQL Vector Store 사용
        collection_name = f"session_{session_id}"
        vectorstore = PGVector.from_documents(
            documents=splits,
            embedding=embeddings,
            collection_name=collection_name,
            connection_string=POSTGRES_CONNECTION_STRING,
            use_jsonb=True  # JSONB 사용으로 성능 향상
        )
        logger.info(f"벡터 저장소 생성 완료 - collection: {collection_name}")
        
        # Retriever 설정: top_k 증가 (4 -> 6)로 더 많은 컨텍스트 검색
        vector_store_mapping[session_id] = vectorstore.as_retriever(
            search_kwargs={"k": 6}
        )
        
        # 페르소나 프롬프트 설정 (범용적, 컨텍스트 기반 답변 강제)
        system_prompts[session_id] = f"""
        당신은 소설 속 인물 '{character_name}'입니다.
        
        **중요 규칙:**
        1. 아래 [Context]에 있는 소설 내용만을 바탕으로 답변하세요.
        2. [Context]에 없는 정보는 절대 만들어내지 마세요.
        3. [Context]에 답변할 수 있는 정보가 없으면 "소설 내용에 그런 정보는 나오지 않습니다" 또는 "모르겠습니다"라고 솔직하게 말하세요.
        4. 컨텍스트 밖의 일반 지식이나 추측을 사용하지 마세요.
        5. 소설에 나오는 인물, 장소, 사건의 이름과 표현을 정확히 사용하세요.
        
        [Context]:
        {{context}}
        """
        logger.debug(f"시스템 프롬프트 설정 완료 - character_name: {character_name}")
        
        # 임시 파일 삭제
        os.remove(file_path)
        logger.info(f"[TRAIN] 완료 - session_id: {session_id}")
        return {"status": "trained", "session_id": session_id}

    except Exception as e:
        logger.error(f"[TRAIN] 오류 발생 - session_id: {session_id}, error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/train-from-s3")
async def train_novel_from_s3(request: TrainFromS3Request):
    """
    S3에서 파일을 다운로드하여 학습
    프론트엔드가 S3에 업로드한 파일을 다운로드하여 RAG 학습 진행
    """
    logger.info(f"[TRAIN-FROM-S3] 시작 - session_id: {request.session_id}, file_key: {request.file_key}, bucket: {request.bucket}")
    file_path = None
    
    try:
        # boto3 임포트 확인
        try:
            import boto3
            from botocore.exceptions import ClientError
        except ImportError:
            logger.error("boto3가 설치되지 않았습니다. pip install boto3를 실행해주세요.")
            raise HTTPException(status_code=500, detail="boto3가 설치되지 않았습니다.")
        
        # S3 클라이언트 생성
        s3_client = boto3.client('s3')
        
        # 임시 파일 경로 생성
        os.makedirs("temp_ai", exist_ok=True)
        filename = os.path.basename(request.file_key) or "novel.txt"
        file_path = f"temp_ai/{request.session_id}_{filename}"
        
        # S3에서 파일 다운로드
        logger.debug(f"S3에서 파일 다운로드 중 - bucket: {request.bucket}, key: {request.file_key}")
        try:
            s3_client.download_file(request.bucket, request.file_key, file_path)
            logger.info(f"파일 다운로드 완료 - {file_path}")
        except ClientError as e:
            logger.error(f"S3 파일 다운로드 실패 - error: {str(e)}")
            raise HTTPException(
                status_code=404, 
                detail=f"S3에서 파일을 찾을 수 없습니다. bucket: {request.bucket}, key: {request.file_key}"
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
        
        # 페르소나 프롬프트 설정
        system_prompts[request.session_id] = f"""
        당신은 소설 속 인물 '{request.character_name}'입니다.
        
        **중요 규칙:**
        1. 아래 [Context]에 있는 소설 내용만을 바탕으로 답변하세요.
        2. [Context]에 없는 정보는 절대 만들어내지 마세요.
        3. [Context]에 답변할 수 있는 정보가 없으면 "소설 내용에 그런 정보는 나오지 않습니다" 또는 "모르겠습니다"라고 솔직하게 말하세요.
        4. 컨텍스트 밖의 일반 지식이나 추측을 사용하지 마세요.
        5. 소설에 나오는 인물, 장소, 사건의 이름과 표현을 정확히 사용하세요.
        
        [Context]:
        {{context}}
        """
        logger.debug(f"시스템 프롬프트 설정 완료 - character_name: {request.character_name}")
        
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

@app.post("/api/ai/chat")
async def chat(request: ChatRequest):
    logger.info(f"[CHAT] 시작 - session_id: {request.session_id}")
    logger.debug(f"사용자 메시지: {request.message[:100]}..." if len(request.message) > 100 else f"사용자 메시지: {request.message}")
    
    session_id = request.session_id
    if session_id not in vector_store_mapping:
        logger.warning(f"[CHAT] 세션 없음 - session_id: {session_id}")
        raise HTTPException(status_code=404, detail="Session not found or expired")

    try:
        retriever = vector_store_mapping[session_id]
        template = system_prompts.get(session_id, "당신은 도움이 되는 챗봇입니다.")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", template),
            ("human", "{question}")
        ])
        
        # 서버의 환경변수 키 사용
        llm = ChatOpenAI(model="gpt-4o", temperature=0.7, openai_api_key=OPENAI_API_KEY)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        logger.debug("RAG 체인 실행 중...")
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

@app.post("/api/ai/update-file")
async def update_novel_file(
    file: UploadFile = File(...),
    session_id: str = Form(...)
):
    """
    파일을 통한 실시간 업데이트 (대량 추가 시 유용)
    """
    logger.info(f"[UPDATE-FILE] 시작 - session_id: {session_id}, filename: {file.filename}")
    file_path = None
    
    try:
        # 파일 임시 저장
        os.makedirs("temp_ai", exist_ok=True)
        file_path = f"temp_ai/{session_id}_update_{file.filename}"
        logger.debug(f"파일 임시 저장 중: {file_path}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

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
        logger.info(f"청킹 완료 - {len(splits)}개 청크 생성")
        
        # 기존 벡터 스토어 로드
        logger.debug("벡터 스토어 로드 중...")
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        collection_name = f"session_{session_id}"
        
        try:
            vectorstore = PGVector(
                collection_name=collection_name,
                connection_string=POSTGRES_CONNECTION_STRING,
                embedding_function=embeddings,
                use_jsonb=True
            )
        except Exception as e:
            logger.warning(f"[UPDATE-FILE] 세션 없음 - session_id: {session_id}, error: {str(e)}")
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(
                status_code=404,
                detail=f"세션이 존재하지 않습니다. 먼저 /api/ai/train으로 초기 학습을 진행해주세요."
            )
        
        # 메타데이터 추가
        logger.debug("메타데이터 추가 중...")
        for doc in splits:
            if not doc.metadata:
                doc.metadata = {}
            doc.metadata["source"] = file.filename
            doc.metadata["added_at"] = str(datetime.now())
        
        # 벡터 스토어에 추가
        logger.debug("벡터 스토어에 문서 추가 중...")
        vectorstore.add_documents(splits)
        
        # 메모리 캐시의 retriever 업데이트
        if session_id in vector_store_mapping:
            vector_store_mapping[session_id] = vectorstore.as_retriever(
                search_kwargs={"k": 6}
            )
            logger.debug("메모리 캐시 retriever 업데이트 완료")
        
        # 임시 파일 삭제
        os.remove(file_path)
        file_path = None
        
        logger.info(f"[UPDATE-FILE] 완료 - session_id: {session_id}, chunks_added: {len(splits)}")
        return {
            "status": "updated",
            "session_id": session_id,
            "chunks_added": len(splits),
            "message": f"파일에서 {len(splits)}개 청크가 성공적으로 추가되었습니다."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        logger.error(f"[UPDATE-FILE] 오류 발생 - session_id: {session_id}, error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"파일 업데이트 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    logger.info("서버 시작 - 포트: 8002")
    # AI 서버 포트: 8002
    uvicorn.run(app, host="0.0.0.0", port=8002)