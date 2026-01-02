# RAGAS 평가 서버 요구사항

## ❌ FastAPI 서버 (ai_server.py)는 필요 없음

RAGAS 평가는 **FastAPI 서버를 거치지 않고** 직접 데이터베이스에 접근합니다.

### 평가 스크립트 동작 방식

```python
# evaluate_ragas.py / run_evaluation.py
# FastAPI 서버를 사용하지 않음!

# 직접 PostgreSQL에 연결
vectorstore = PGVector(
    collection_name=f"session_{session_id}",
    connection_string=POSTGRES_CONNECTION_STRING,  # 직접 DB 연결
    embedding_function=embeddings,
    use_jsonb=True
)

# RAG 체인 직접 생성
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

## ✅ 필요한 것

### 1. PostgreSQL 데이터베이스 실행 중
- 벡터 스토어가 저장된 PostgreSQL이 실행 중이어야 함
- `POSTGRES_CONNECTION_STRING` 환경변수가 올바르게 설정되어 있어야 함

### 2. 환경변수 설정
```bash
# .env 파일에 필요
OPENAI_API_KEY=your_key_here
POSTGRES_CONNECTION_STRING=postgresql://user:password@localhost:5432/dbname
```

### 3. 평가 스크립트 실행
```bash
# 서버 없이 바로 실행 가능
python evaluate_ragas.py --session_id sess_123 --dataset evaluation_dataset.json
```

## 🔄 서버가 필요한 경우

### FastAPI 서버 (ai_server.py)가 필요한 경우:
- ✅ 프론트엔드에서 채팅할 때
- ✅ 소설 파일 업로드 및 학습할 때
- ❌ RAGAS 평가할 때는 **불필요**

### PostgreSQL이 필요한 경우:
- ✅ 소설 학습 시 (벡터 스토어 저장)
- ✅ 채팅 시 (벡터 검색)
- ✅ **RAGAS 평가 시 (벡터 스토어 읽기)**

## 📊 비교

| 작업 | FastAPI 서버 | PostgreSQL | 평가 스크립트 |
|------|-------------|------------|--------------|
| 소설 업로드/학습 | ✅ 필요 | ✅ 필요 | ❌ 불필요 |
| 채팅 | ✅ 필요 | ✅ 필요 | ❌ 불필요 |
| RAGAS 평가 | ❌ 불필요 | ✅ 필요 | ✅ 필요 |

## 💡 요약

**RAGAS 평가를 하려면:**
1. ❌ FastAPI 서버 (ai_server.py) - **켜져있을 필요 없음**
2. ✅ PostgreSQL 데이터베이스 - **반드시 실행 중이어야 함**
3. ✅ 평가 스크립트 (evaluate_ragas.py) - **실행 필요**

**평가 스크립트는 독립적으로 실행되며, 서버를 거치지 않고 직접 데이터베이스에 접근합니다.**


















