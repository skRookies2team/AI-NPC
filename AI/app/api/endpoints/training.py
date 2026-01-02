import os
from fastapi import HTTPException
from app.models.requests import TrainFromS3Request, TrainTextRequest
from app.services.training import (
    download_from_s3,
    load_and_chunk_text,
    save_text_to_temp_file,
    cleanup_temp_file
)
from app.services.vectorstore import create_vectorstore
from app.services.character import generate_system_prompt
from app.core.state import set_system_prompt
from app.database.session import save_session_info
from app.core.logging import logger
from app.config import settings
import traceback

async def train_novel_from_s3(request: TrainFromS3Request):
    """
    S3에서 파일을 다운로드하여 학습
    프론트엔드가 S3에 업로드한 파일을 다운로드하여 RAG 학습 진행
    """
    logger.info(f"[TRAIN-FROM-S3] 시작 - session_id: {request.session_id}, file_key: {request.file_key}, bucket: {settings.S3_BUCKET}")
    file_path = None

    try:
        # boto3 임포트 확인
        try:
            import boto3
            from botocore.exceptions import ClientError, NoCredentialsError
        except ImportError:
            logger.error("boto3가 설치되지 않았습니다. pip install boto3를 실행해주세요.")
            raise HTTPException(status_code=500, detail="boto3가 설치되지 않았습니다.")

        # 임시 파일 경로 생성
        os.makedirs("temp_ai", exist_ok=True)
        filename = os.path.basename(request.file_key) or "novel.txt"
        file_path = f"temp_ai/{request.session_id}_{filename}"

        # S3에서 파일 다운로드
        logger.debug(f"S3에서 파일 다운로드 중 - bucket: {settings.S3_BUCKET}, key: {request.file_key}")
        download_from_s3(request.file_key, file_path)

        # 텍스트 로드 및 청킹
        logger.debug("텍스트 파일 로드 및 청킹 중...")
        splits, total_chars = load_and_chunk_text(file_path)

        # 임베딩 및 벡터 저장
        logger.debug("임베딩 생성 및 벡터 저장소 생성 중...")
        create_vectorstore(request.session_id, splits)

        # 기본 캐릭터 설정
        character_name = request.character_name.strip() if request.character_name else "캐릭터"
        system_prompt = generate_system_prompt(character_name, "")

        # 메모리와 데이터베이스에 저장
        set_system_prompt(request.session_id, character_name, system_prompt)
        save_session_info(
            session_id=request.session_id,
            character_name=character_name,
            system_prompt=system_prompt,
            character_description=None
        )

        # 임시 파일 삭제
        cleanup_temp_file(file_path)

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
        cleanup_temp_file(file_path)
        raise HTTPException(status_code=500, detail=f"학습 중 오류 발생: {str(e)}")

async def train_novel_from_text(request: TrainTextRequest):
    """
    테스트용: 텍스트를 직접 받아서 학습
    실제 운영에서는 /api/ai/train-from-s3를 사용하세요
    """
    logger.info(f"[TRAIN-TEXT] 시작 - session_id: {request.session_id}")
    file_path = None

    try:
        # 임시 파일로 저장
        file_path = save_text_to_temp_file(request.session_id, request.content)

        # 텍스트 로드 및 청킹
        splits, total_chars = load_and_chunk_text(file_path)

        # 임베딩 및 벡터 저장
        create_vectorstore(request.session_id, splits)

        # 캐릭터 이름 처리
        character_name = request.character_name.strip() if request.character_name else "캐릭터"
        system_prompt = generate_system_prompt(character_name, "")

        # train-text는 소설만 학습 (character_name은 나중에 /api/ai/character에서 설정)
        logger.info(f"[TRAIN-TEXT] 소설 학습 완료 - session_id: {request.session_id}")
        logger.info(f"[TRAIN-TEXT] 캐릭터는 /api/ai/character 엔드포인트에서 설정하세요.")

        # 임시 파일 삭제
        cleanup_temp_file(file_path)

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
        cleanup_temp_file(file_path)
        raise HTTPException(status_code=500, detail=f"학습 중 오류 발생: {str(e)}")
