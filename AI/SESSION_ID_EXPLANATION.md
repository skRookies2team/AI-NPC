# 세션 ID (Session ID) 설명

## 📋 세션 ID란?

**세션 ID**는 소설 파일을 업로드하고 학습할 때 생성되는 **고유 식별자**입니다. 각 소설/캐릭터를 구분하는 데 사용됩니다.

## 🔄 생성 과정

### 1. 소설 업로드 시 생성
```javascript
// test-backend/server.js (26줄)
const session_id = 'sess_' + Date.now();
```

**형식**: `sess_1764661118623`
- `sess_` 접두사
- `Date.now()` 타임스탬프 (밀리초 단위)

**예시**:
- `sess_1764660940439`
- `sess_1764661041727`
- `sess_1764661118623`

### 2. 학습 완료 후 반환
```python
# ai_server.py (85줄)
return {"status": "trained", "session_id": session_id}
```

프론트엔드에서 이 세션 ID를 받아서 저장하고, 이후 채팅할 때 사용합니다.

## 💾 저장 위치

### 1. PostgreSQL 벡터 스토어
```python
# ai_server.py (62줄)
collection_name = f"session_{session_id}"
# 예: "session_sess_1764661118623"
```

PostgreSQL에 `session_{session_id}` 형식의 컬렉션으로 벡터 스토어가 저장됩니다.

### 2. 서버 메모리
```python
# ai_server.py (30-31줄)
vector_store_mapping = {}  # 세션 ID -> Retriever 매핑
system_prompts = {}        # 세션 ID -> 프롬프트 매핑
```

서버가 실행 중일 때만 메모리에 저장됩니다.

## 🎯 사용 목적

### 1. 채팅 시 사용
```python
# ai_server.py (93-98줄)
session_id = request.session_id
if session_id not in vector_store_mapping:
    raise HTTPException(status_code=404, detail="Session not found or expired")

retriever = vector_store_mapping[session_id]
template = system_prompts.get(session_id, "...")
```

특정 세션의 벡터 스토어와 프롬프트를 가져와서 사용합니다.

### 2. RAGAS 평가 시 사용
```python
# evaluate_ragas.py (74줄)
collection_name = f"session_{session_id}"

vectorstore = PGVector(
    collection_name=collection_name,
    connection_string=POSTGRES_CONNECTION_STRING,
    ...
)
```

평가할 세션의 벡터 스토어를 로드합니다.

## 📝 세션 ID 확인 방법

### 1. 프론트엔드에서 확인
소설을 업로드하면 응답으로 세션 ID를 받습니다:
```javascript
// test-frontend/index.html (86줄)
currentSessionId = data.session_id;
```

### 2. 백엔드 로그 확인
학습 완료 시 로그에 세션 ID가 출력됩니다.

### 3. PostgreSQL에서 확인
```sql
-- PostgreSQL에서 세션 목록 확인
SELECT DISTINCT collection_name 
FROM langchain_pg_collection 
WHERE collection_name LIKE 'session_%';
```

### 4. temp_ai 폴더 확인
```bash
# AI/temp_ai 폴더의 파일명에서 확인 가능
ls AI/temp_ai/
# sess_1764660940439_파일명.txt
# sess_1764661041727_파일명.txt
```

## ⚠️ 주의사항

### 1. 세션 만료
- 서버를 재시작하면 메모리의 `vector_store_mapping`이 초기화됩니다
- 하지만 PostgreSQL의 벡터 스토어는 유지됩니다
- PostgreSQL에서 다시 로드하면 사용 가능합니다

### 2. 세션 ID 형식
- 반드시 `sess_` 접두사가 있어야 합니다
- 숫자만으로는 작동하지 않습니다

### 3. RAGAS 평가 시
```bash
# 올바른 사용
python evaluate_ragas.py --session_id sess_1764661118623 --dataset evaluation_dataset.json

# 잘못된 사용
python evaluate_ragas.py --session_id 1764661118623 --dataset evaluation_dataset.json  # ❌
```

## 🔍 실제 예시

### 세션 생성 과정
1. 사용자가 소설 파일 업로드
2. 백엔드에서 `sess_1764661118623` 생성
3. PostgreSQL에 `session_sess_1764661118623` 컬렉션 생성
4. 벡터 스토어에 소설 내용 저장
5. 프론트엔드에 세션 ID 반환

### 채팅 시 사용
```javascript
// 프론트엔드
fetch('/chat', {
    method: 'POST',
    body: JSON.stringify({
        session_id: 'sess_1764661118623',
        message: '안녕하세요'
    })
})
```

### 평가 시 사용
```bash
python evaluate_ragas.py \
    --session_id sess_1764661118623 \
    --dataset evaluation_dataset.json \
    --character_name "홍길동"
```

## 💡 요약

- **세션 ID**: 각 소설/캐릭터를 구분하는 고유 식별자
- **형식**: `sess_` + 타임스탬프
- **저장**: PostgreSQL 벡터 스토어 (`session_{session_id}`)
- **사용**: 채팅, 평가 등 모든 작업에서 해당 세션을 식별




