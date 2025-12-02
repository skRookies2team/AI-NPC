import os
import shutil
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
        
        # 텍스트 청킹 (Chunking)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
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
        
        vector_store_mapping[session_id] = vectorstore.as_retriever()
        
        # 페르소나 프롬프트 설정
        system_prompts[session_id] = f"""
        당신은 소설 속 인물 '{character_name}'입니다.
        아래 [Context]에 있는 소설 내용을 바탕으로 사용자와 대화하세요.
        소설에 나오지 않는 내용은 캐릭터의 성격에 맞춰 자연스럽게 지어내세요.
        
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

if __name__ == "__main__":
    import uvicorn
    # AI 서버 포트: 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)