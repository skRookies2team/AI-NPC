from typing import Optional, List, Dict
import psycopg2
from psycopg2.extras import RealDictCursor
from app.database.connection import get_db_connection
from app.core.logging import logger

def init_session_table():
    """세션 정보 저장 테이블 생성 (한 세션에 여러 캐릭터 지원)"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # 기존 테이블이 있는지 확인
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'session_info'
            );
        """)
        table_exists = cur.fetchone()[0]

        if table_exists:
            # 기존 테이블이 있으면 PRIMARY KEY 변경 시도
            try:
                # 기존 PRIMARY KEY 제약조건 확인 및 제거
                cur.execute("""
                    SELECT constraint_name
                    FROM information_schema.table_constraints
                    WHERE table_name = 'session_info'
                    AND constraint_type = 'PRIMARY KEY';
                """)
                pk_constraint = cur.fetchone()
                if pk_constraint:
                    cur.execute(f"ALTER TABLE session_info DROP CONSTRAINT {pk_constraint[0]};")
                    logger.info(f"기존 PRIMARY KEY 제약조건 제거: {pk_constraint[0]}")

                # 복합 PRIMARY KEY 추가
                cur.execute("""
                    ALTER TABLE session_info
                    ADD PRIMARY KEY (session_id, character_name);
                """)
                logger.info("기존 테이블에 복합 PRIMARY KEY 추가 완료")
            except Exception as e:
                # 이미 복합 키가 있거나 다른 이유로 실패할 수 있음
                logger.debug(f"PRIMARY KEY 변경 시도 중 (이미 변경되었을 수 있음): {str(e)}")
                conn.rollback()
        else:
            # 새 테이블 생성
            cur.execute("""
                CREATE TABLE session_info (
                    session_id VARCHAR(255) NOT NULL,
                    character_name VARCHAR(255) NOT NULL,
                    system_prompt TEXT NOT NULL,
                    character_description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (session_id, character_name)
                )
            """)
            logger.info("새 세션 정보 테이블 생성 완료 (복합 PRIMARY KEY)")

        conn.commit()
        cur.close()
        conn.close()
        logger.info("세션 정보 테이블 초기화 완료")
    except Exception as e:
        logger.error(f"세션 정보 테이블 초기화 실패: {str(e)}")
        raise

def save_session_info(session_id: str, character_name: str, system_prompt: str, character_description: str = None):
    """세션 정보를 PostgreSQL에 저장 (한 세션에 여러 캐릭터 지원)"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO session_info (session_id, character_name, system_prompt, character_description, updated_at)
            VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (session_id, character_name)
            DO UPDATE SET
                system_prompt = EXCLUDED.system_prompt,
                character_description = EXCLUDED.character_description,
                updated_at = CURRENT_TIMESTAMP
        """, (session_id, character_name, system_prompt, character_description))
        conn.commit()
        cur.close()
        conn.close()
        logger.debug(f"세션 정보 저장 완료 - session_id: {session_id}, character_name: {character_name}")
    except Exception as e:
        logger.error(f"세션 정보 저장 실패 - session_id: {session_id}, character_name: {character_name}, error: {str(e)}")
        raise

def load_session_info(session_id: str, character_name: str = None) -> Optional[Dict]:
    """PostgreSQL에서 세션 정보 로드 (특정 캐릭터 또는 세션의 모든 캐릭터)"""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        if character_name:
            # 특정 캐릭터만 로드
            cur.execute("""
                SELECT session_id, character_name, system_prompt, character_description, created_at, updated_at
                FROM session_info
                WHERE session_id = %s AND character_name = %s
            """, (session_id, character_name))
            result = cur.fetchone()
            cur.close()
            conn.close()
            if result:
                logger.debug(f"세션 정보 로드 완료 - session_id: {session_id}, character_name: {character_name}")
                return dict(result)
            return None
        else:
            # 세션의 모든 캐릭터 로드 (첫 번째 캐릭터 반환 - 하위 호환성)
            cur.execute("""
                SELECT session_id, character_name, system_prompt, character_description, created_at, updated_at
                FROM session_info
                WHERE session_id = %s
                ORDER BY updated_at DESC
                LIMIT 1
            """, (session_id,))
            result = cur.fetchone()
            cur.close()
            conn.close()
            if result:
                logger.debug(f"세션 정보 로드 완료 - session_id: {session_id} (첫 번째 캐릭터)")
                return dict(result)
            return None
    except Exception as e:
        logger.warning(f"세션 정보 로드 실패 - session_id: {session_id}, character_name: {character_name}, error: {str(e)}")
        return None

def load_all_characters_for_session(session_id: str) -> List[Dict]:
    """세션의 모든 캐릭터 정보 로드"""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT session_id, character_name, system_prompt, character_description, created_at, updated_at
            FROM session_info
            WHERE session_id = %s
            ORDER BY updated_at DESC
        """, (session_id,))
        results = cur.fetchall()
        cur.close()
        conn.close()
        characters = [dict(row) for row in results]
        logger.debug(f"세션의 모든 캐릭터 로드 완료 - session_id: {session_id}, 캐릭터 수: {len(characters)}")
        return characters
    except Exception as e:
        logger.warning(f"세션의 모든 캐릭터 로드 실패 - session_id: {session_id}, error: {str(e)}")
        return []

def list_all_sessions() -> List[Dict]:
    """PostgreSQL에서 모든 세션 목록 조회"""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT session_id, character_name, character_description, created_at, updated_at
            FROM session_info
            ORDER BY updated_at DESC
        """)
        results = cur.fetchall()
        cur.close()
        conn.close()
        sessions = [dict(row) for row in results]
        logger.debug(f"세션 목록 조회 완료 - 총 {len(sessions)}개 세션")
        return sessions
    except Exception as e:
        logger.error(f"세션 목록 조회 실패 - error: {str(e)}")
        return []
