from pydantic import BaseModel

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

class TrainTextRequest(BaseModel):
    """테스트용: 텍스트 직접 전송하여 학습"""
    session_id: str
    content: str  # 소설 텍스트 내용
    character_name: str = ""  # 캐릭터 이름 (선택적, 나중에 /api/ai/character에서 설정 가능)
