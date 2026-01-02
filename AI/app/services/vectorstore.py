from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from typing import List, Optional
from app.config import settings
from app.core.state import set_retriever, get_retriever
from app.core.logging import logger

def create_vectorstore(session_id: str, documents: List[Document]):
    """세션을 위한 새 벡터 스토어 생성"""
    embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)
    collection_name = f"session_{session_id}"

    vectorstore = PGVector.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        connection_string=settings.POSTGRES_CONNECTION_STRING,
        use_jsonb=True
    )
    logger.info(f"벡터 저장소 생성 완료 - collection: {collection_name}")

    retriever = vectorstore.as_retriever(search_kwargs={"k": settings.RETRIEVER_K})
    set_retriever(session_id, retriever)

    return vectorstore

def load_vectorstore(session_id: str) -> Optional[PGVector]:
    """PostgreSQL에서 기존 벡터 스토어 로드"""
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)
        collection_name = f"session_{session_id}"

        vectorstore = PGVector(
            collection_name=collection_name,
            connection_string=settings.POSTGRES_CONNECTION_STRING,
            embedding_function=embeddings,
            use_jsonb=True
        )

        # 벡터 스토어가 데이터를 가지고 있는지 확인
        test_docs = vectorstore.similarity_search("test", k=1)
        if len(test_docs) > 0:
            retriever = vectorstore.as_retriever(search_kwargs={"k": settings.RETRIEVER_K})
            set_retriever(session_id, retriever)
            logger.info(f"PostgreSQL에서 벡터 스토어 로드 완료 - collection: {collection_name}")
            return vectorstore
    except Exception as e:
        logger.debug(f"{session_id}에 대한 벡터 스토어 로드 실패: {e}")

    return None

def add_documents_to_vectorstore(session_id: str, documents: List[Document]) -> int:
    """기존 벡터 스토어에 문서 추가"""
    embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)
    collection_name = f"session_{session_id}"

    vectorstore = PGVector(
        collection_name=collection_name,
        connection_string=settings.POSTGRES_CONNECTION_STRING,
        embedding_function=embeddings,
        use_jsonb=True
    )

    vectorstore.add_documents(documents)
    logger.info(f"벡터 스토어에 {len(documents)}개 문서 추가 완료")

    # 인메모리 retriever 업데이트
    retriever = vectorstore.as_retriever(search_kwargs={"k": settings.RETRIEVER_K})
    set_retriever(session_id, retriever)

    return len(documents)
