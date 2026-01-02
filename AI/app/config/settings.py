import os
from dotenv import load_dotenv

# 환경변수 로드 (.env 파일에서 API 키 가져옴)
load_dotenv()

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# PostgreSQL
POSTGRES_CONNECTION_STRING = os.getenv("POSTGRES_CONNECTION_STRING")

# AWS 자격 증명 (환경변수에서 가져오기)
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_S3_REGION", "ap-northeast-2")
S3_BUCKET = os.getenv("AWS_S3_BUCKET")  # S3 버킷 이름 (환경변수에서만 가져옴)

# Server Configuration
HOST = "0.0.0.0"
PORT = 8002

# RAG Configuration
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
RETRIEVER_K = 6
TEMPERATURE = 0.8
MODEL_NAME = "gpt-4o"

# Logging Configuration
LOG_DIR = "logs"
LOG_FILE = "ai_server.log"

# 필수 환경변수 검증
if not S3_BUCKET:
    raise ValueError(".env 파일에 AWS_S3_BUCKET 또는 S3_BUCKET이 설정되지 않았습니다.")

if not OPENAI_API_KEY:
    raise ValueError(".env 파일에 OPENAI_API_KEY가 설정되지 않았습니다.")

if not POSTGRES_CONNECTION_STRING:
    raise ValueError(".env 파일에 POSTGRES_CONNECTION_STRING이 설정되지 않았습니다.")
