from datetime import datetime
from fastapi import HTTPException
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.models.requests import UpdateContentRequest
from app.services.vectorstore import add_documents_to_vectorstore
from app.config import settings
from app.core.logging import logger
import traceback

async def update_novel_content(request: UpdateContentRequest):
    """
    실시간으로 변화하는 소설 상황을 RAG에 추가
    게임 진행 중 발생하는 새로운 이벤트나 스토리 내용을 벡터 DB에 추가하여 학습
    """
    logger.info(f"[UPDATE] 시작 - session_id: {request.session_id}")
    logger.debug(f"추가할 내용 길이: {len(request.content)} 문자")

    try:
        # 빈 내용 확인
        new_content = request.content.strip()
        if not new_content:
            logger.warning(f"[UPDATE] 빈 내용 - session_id: {request.session_id}")
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
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        splits = text_splitter.split_documents([new_doc])
        logger.info(f"청킹 완료 - {len(splits)}개 청크 생성")

        # 벡터 스토어에 새 문서 추가
        logger.debug("벡터 스토어에 문서 추가 중...")
        chunks_added = add_documents_to_vectorstore(request.session_id, splits)

        logger.info(f"[UPDATE] 완료 - session_id: {request.session_id}, chunks_added: {chunks_added}")
        return {
            "status": "updated",
            "session_id": request.session_id,
            "chunks_added": chunks_added,
            "message": f"새로운 소설 내용 {chunks_added}개 청크가 성공적으로 추가되었습니다."
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[UPDATE] 오류 발생 - session_id: {request.session_id}, error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"내용 추가 중 오류 발생: {str(e)}")
