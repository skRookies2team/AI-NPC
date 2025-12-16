import os
import shutil
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
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

app = FastAPI()

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

@app.post("/api/ai/train")
async def train_novel(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    character_name: str = Form(...)
):
    try:
        # 파일 임시 저장
        os.makedirs("temp_ai", exist_ok=True)
        file_path = f"temp_ai/{session_id}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 텍스트 로드
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        
        # 텍스트 청킹 (Chunking) - 최적화된 크기
        # chunk_size: 800 (더 작은 청크로 정확도 향상)
        # chunk_overlap: 150 (문맥 유지를 위한 적절한 오버랩)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""]  # 문단 > 문장 > 단어 순서로 분할
        )
        splits = text_splitter.split_documents(docs)

        # 임베딩 및 벡터 저장 (서버의 환경변수 키 사용)
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
        
        # 임시 파일 삭제
        os.remove(file_path)
        return {"status": "trained", "session_id": session_id}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/chat")
async def chat(request: ChatRequest):
    session_id = request.session_id
    if session_id not in vector_store_mapping:
        raise HTTPException(status_code=404, detail="Session not found or expired")

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

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = rag_chain.invoke(request.message)
    return {"reply": response}

@app.post("/api/ai/update")
async def update_novel_content(request: UpdateContentRequest):
    """
    실시간으로 변화하는 소설 상황을 RAG에 추가
    게임 진행 중 발생하는 새로운 이벤트나 스토리 내용을 벡터 DB에 추가하여 학습
    """
    try:
        session_id = request.session_id
        
        # 기존 벡터 스토어 로드
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
            raise HTTPException(
                status_code=404, 
                detail=f"세션이 존재하지 않습니다. 먼저 /api/ai/train으로 초기 학습을 진행해주세요. Error: {str(e)}"
            )
        
        # 새로 추가할 텍스트를 문서로 변환
        new_content = request.content.strip()
        if not new_content:
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
        
        # 텍스트 청킹 (기존과 동일한 설정 사용)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        splits = text_splitter.split_documents([new_doc])
        
        # 벡터 스토어에 새 문서 추가
        vectorstore.add_documents(splits)
        
        # 메모리 캐시의 retriever도 업데이트 (존재하는 경우)
        if session_id in vector_store_mapping:
            vector_store_mapping[session_id] = vectorstore.as_retriever(
                search_kwargs={"k": 6}
            )
        
        return {
            "status": "updated",
            "session_id": session_id,
            "chunks_added": len(splits),
            "message": f"새로운 소설 내용 {len(splits)}개 청크가 성공적으로 추가되었습니다."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"내용 추가 중 오류 발생: {str(e)}")

@app.post("/api/ai/update-file")
async def update_novel_file(
    file: UploadFile = File(...),
    session_id: str = Form(...)
):
    """
    파일을 통한 실시간 업데이트 (대량 추가 시 유용)
    """
    try:
        # 파일 임시 저장
        os.makedirs("temp_ai", exist_ok=True)
        file_path = f"temp_ai/{session_id}_update_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 텍스트 로드
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        
        # 텍스트 청킹
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        splits = text_splitter.split_documents(docs)
        
        # 기존 벡터 스토어 로드
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
            os.remove(file_path)
            raise HTTPException(
                status_code=404,
                detail=f"세션이 존재하지 않습니다. 먼저 /api/ai/train으로 초기 학습을 진행해주세요."
            )
        
        # 메타데이터 추가
        for doc in splits:
            if not doc.metadata:
                doc.metadata = {}
            doc.metadata["source"] = file.filename
            doc.metadata["added_at"] = str(__import__("datetime").datetime.now())
        
        # 벡터 스토어에 추가
        vectorstore.add_documents(splits)
        
        # 메모리 캐시의 retriever 업데이트
        if session_id in vector_store_mapping:
            vector_store_mapping[session_id] = vectorstore.as_retriever(
                search_kwargs={"k": 6}
            )
        
        # 임시 파일 삭제
        os.remove(file_path)
        
        return {
            "status": "updated",
            "session_id": session_id,
            "chunks_added": len(splits),
            "message": f"파일에서 {len(splits)}개 청크가 성공적으로 추가되었습니다."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"파일 업데이트 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # AI 서버 포트: 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)