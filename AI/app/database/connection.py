import psycopg2
from urllib.parse import urlparse
from app.config import settings

def get_db_connection():
    """PostgreSQL 연결 문자열에서 연결 객체 생성"""
    # connection_string 형식: "postgresql://user:password@host:port/database"
    # psycopg2는 "postgresql://" 대신 키워드 인자나 직접 파싱 필요
    # URL 파싱
    parsed = urlparse(settings.POSTGRES_CONNECTION_STRING)
    return psycopg2.connect(
        host=parsed.hostname,
        port=parsed.port or 5432,
        database=parsed.path[1:] if parsed.path else None,  # '/' 제거
        user=parsed.username,
        password=parsed.password
    )
