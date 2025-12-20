# RAGAS 평가 가이드

RAGAS(RAG Assessment)는 RAG(Retrieval-Augmented Generation) 시스템의 성능을 평가하기 위한 프레임워크입니다.

## 설치

```bash
cd AI
pip install -r requirements.txt
```

또는 직접 설치:
```bash
pip install ragas datasets
```

## 평가 메트릭 설명

RAGAS는 다음 4가지 주요 메트릭을 제공합니다:

1. **Faithfulness (정확성)**: 답변이 검색된 컨텍스트에 기반하여 정확한지 평가 (0-1)
   - 높을수록 답변이 컨텍스트에 충실함
   - 낮으면 환각(hallucination) 가능성

2. **Answer Relevancy (답변 관련성)**: 답변이 질문과 얼마나 관련이 있는지 평가 (0-1)
   - 높을수록 질문에 대한 답변이 적절함

3. **Context Precision (컨텍스트 정밀도)**: 검색된 컨텍스트가 질문과 얼마나 관련이 있는지 평가 (0-1)
   - 높을수록 검색된 문서가 질문과 관련이 있음

4. **Context Recall (컨텍스트 재현율)**: 검색된 컨텍스트가 실제 필요한 정보를 얼마나 포함하는지 평가 (0-1)
   - 높을수록 필요한 정보를 잘 검색함
   - Ground truth가 필요함

## 사용 방법

### 1. 평가 데이터셋 준비

평가 데이터셋은 JSON 형식으로 준비합니다:

```json
[
  {
    "question": "캐릭터의 이름은 무엇인가요?",
    "ground_truth": "캐릭터의 이름은 홍길동입니다.",
    "ground_truths": ["홍길동", "길동"]  // 선택사항, 여러 정답 가능
  },
  {
    "question": "캐릭터의 성격은 어떤가요?",
    "ground_truth": "캐릭터는 밝고 활발한 성격입니다."
  }
]
```

**필수 필드:**
- `question`: 평가할 질문

**선택 필드:**
- `ground_truth`: 단일 정답 (문자열)
- `ground_truths`: 여러 정답 (배열) - Context Recall 계산에 사용

### 2. 샘플 데이터셋 생성

```bash
python evaluate_ragas.py --create_sample --dataset evaluation_dataset.json
```

생성된 파일을 수정하여 실제 평가 데이터로 사용하세요.

### 3. 평가 실행

```bash
python evaluate_ragas.py --session_id <세션ID> --dataset evaluation_dataset.json
```

**옵션:**
- `--session_id`: 평가할 세션 ID (필수)
- `--dataset`: 평가 데이터셋 파일 경로 (기본값: evaluation_dataset.json)
- `--character_name`: 캐릭터 이름 (기본값: 캐릭터)
- `--k`: 검색할 문서 개수 (기본값: 4)

**예시:**
```bash
python evaluate_ragas.py --session_id sess_123 --dataset my_eval_data.json --character_name "홍길동" --k 5
```

### 4. 결과 확인

평가가 완료되면:
1. 콘솔에 메트릭 점수가 출력됩니다
2. `evaluation_results_<session_id>.json` 파일에 상세 결과가 저장됩니다

**결과 파일 구조:**
```json
{
  "session_id": "sess_123",
  "num_questions": 10,
  "metrics": {
    "faithfulness": 0.85,
    "answer_relevancy": 0.90,
    "context_precision": 0.80,
    "context_recall": 0.75,
    "average": 0.825
  },
  "detailed_results": [
    {
      "question": "...",
      "answer": "...",
      "contexts": [...],
      "faithfulness": 0.85,
      "answer_relevancy": 0.90,
      ...
    }
  ]
}
```

## 평가 데이터셋 작성 팁

1. **다양한 질문 유형 포함**
   - 사실 확인 질문
   - 성격/특성 질문
   - 사건/상황 질문
   - 추론이 필요한 질문

2. **Ground Truth 작성**
   - 정확한 답변을 제공하세요
   - 여러 정답이 가능한 경우 `ground_truths` 배열 사용
   - Context Recall을 정확히 측정하려면 관련 문서의 핵심 내용을 포함

3. **질문 개수**
   - 최소 10개 이상 권장
   - 더 많은 질문일수록 평가 결과가 신뢰할 수 있음

## 메트릭 해석

### 좋은 점수 기준
- **Faithfulness**: 0.8 이상
- **Answer Relevancy**: 0.8 이상
- **Context Precision**: 0.7 이상
- **Context Recall**: 0.7 이상 (Ground truth가 있을 때)

### 문제 해결

**Faithfulness가 낮은 경우:**
- 답변이 컨텍스트를 벗어나거나 환각이 발생
- 프롬프트 개선 필요
- LLM temperature 조정

**Answer Relevancy가 낮은 경우:**
- 답변이 질문과 관련이 없음
- 프롬프트에서 질문에 집중하도록 개선

**Context Precision이 낮은 경우:**
- 검색된 문서가 질문과 관련이 없음
- 임베딩 모델 개선
- 검색 전략 변경 (예: reranking)

**Context Recall이 낮은 경우:**
- 필요한 정보를 검색하지 못함
- 청킹 전략 개선
- 검색 개수(k) 증가
- 임베딩 모델 개선

## 주의사항

1. **API 비용**: RAGAS 평가는 LLM을 사용하므로 API 비용이 발생합니다
2. **실행 시간**: 질문이 많을수록 평가 시간이 오래 걸립니다
3. **Ground Truth**: Context Recall을 측정하려면 정확한 Ground Truth가 필요합니다

## 참고 자료

- [RAGAS 공식 문서](https://docs.ragas.io/)
- [RAGAS GitHub](https://github.com/explodinggradients/ragas)









