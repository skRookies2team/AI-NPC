import os
from typing import List, Tuple
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.config import settings
from app.core.logging import logger

def download_from_s3(file_key: str, local_path: str):
    """S3에서 파일을 로컬 경로로 다운로드"""
    import boto3
    from botocore.exceptions import ClientError

    if not settings.AWS_ACCESS_KEY_ID or not settings.AWS_SECRET_ACCESS_KEY:
        raise ValueError("AWS 자격 증명이 설정되지 않았습니다.")

    s3_client = boto3.client(
        's3',
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_REGION
    )

    try:
        s3_client.download_file(settings.S3_BUCKET, file_key, local_path)
        logger.info(f"파일 다운로드 완료 - {local_path}")
    except ClientError as e:
        logger.error(f"S3 파일 다운로드 실패 - error: {str(e)}")
        raise

def load_and_chunk_text(file_path: str) -> Tuple[List[Document], int]:
    """텍스트 파일 로드 및 청킹"""
    # 문서 로드
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()

    total_chars = sum(len(doc.page_content) for doc in docs)
    logger.info(f"로드된 문서 수: {len(docs)}, 총 문자 수: {total_chars}")

    # 청킹
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    logger.info(f"청킹 완료 - 총 {len(splits)}개 청크 생성")

    return splits, total_chars

def save_text_to_temp_file(session_id: str, content: str, filename: str = "novel.txt") -> str:
    """텍스트 콘텐츠를 임시 파일로 저장"""
    os.makedirs("temp_ai", exist_ok=True)
    file_path = f"temp_ai/{session_id}_{filename}"

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    logger.info(f"임시 파일 저장 완료 - {file_path}")
    return file_path

def cleanup_temp_file(file_path: str):
    """임시 파일 제거"""
    if file_path and os.path.exists(file_path):
        os.remove(file_path)
        logger.debug(f"임시 파일 삭제 완료 - {file_path}")
