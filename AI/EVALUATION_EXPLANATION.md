# Questions와 Ground Truth 평가 방식 설명

## 📋 데이터 구조

### 1. **Questions (질문)**
평가할 질문들입니다. RAG 시스템에 입력되는 사용자 질문입니다.

```python
questions = [
    "캐릭터의 이름은 무엇인가요?",
    "캐릭터의 성격은 어떤가요?",
    "주요 사건은 무엇이었나요?",
]
```

**정의 방법:**
- JSON 파일에서 로드 (`evaluation_dataset.json`)
- 또는 코드에서 직접 정의
- 실제 소설 내용을 기반으로 한 질문이어야 함

### 2. **Ground Truth (정답)**
각 질문에 대한 **정확한 답변**입니다. 시스템이 생성한 답변과 비교하는 기준입니다.

```python
ground_truths = [
    ["캐릭터의 이름은 홍길동입니다."],  # 단일 정답
    ["캐릭터는 밝고 활발한 성격입니다.", "캐릭터는 긍정적인 성격입니다."],  # 여러 정답 가능
]
```

**정의 방법:**
- `ground_truth`: 단일 정답 (문자열)
- `ground_truths`: 여러 정답 (배열) - 더 정확한 평가를 위해

## 🔄 평가 프로세스

### Step 1: 질문 입력
```python
question = "캐릭터의 이름은 무엇인가요?"
```

### Step 2: RAG 시스템 실행
```python
# 1. Retriever가 관련 컨텍스트 검색
contexts = retriever.get_relevant_documents(question, k=4)
# 결과: ["소설의 첫 부분...", "캐릭터 소개 부분...", ...]

# 2. LLM이 컨텍스트를 바탕으로 답변 생성
answer = rag_chain.invoke(question)
# 결과: "캐릭터의 이름은 홍길동입니다."
```

### Step 3: 평가 데이터셋 구성
```python
evaluation_dataset = Dataset.from_dict({
    "question": ["캐릭터의 이름은 무엇인가요?"],
    "answer": ["캐릭터의 이름은 홍길동입니다."],  # RAG 시스템이 생성한 답변
    "contexts": [["소설의 첫 부분...", "캐릭터 소개 부분..."]],  # 검색된 문서들
    "ground_truths": [["캐릭터의 이름은 홍길동입니다."]],  # 정답
    "reference": ["캐릭터의 이름은 홍길동입니다."],  # context_precision용
})
```

## 📊 각 메트릭이 Questions와 Ground Truth를 어떻게 사용하는가?

### 1. **Faithfulness (정확성)**
- **사용 데이터**: `question`, `answer`, `contexts`
- **평가 방식**: 
  - 답변이 검색된 컨텍스트에 기반한 정확한 정보인지 확인
  - Ground Truth는 **사용하지 않음** (컨텍스트 기반 평가)
  - 환각(hallucination) 감지

**예시:**
```
Question: "캐릭터의 이름은 무엇인가요?"
Contexts: ["홍길동은 주인공이다", "이야기는 서울에서 시작된다"]
Answer: "홍길동입니다"  ✅ Faithfulness 높음 (컨텍스트 기반)
Answer: "김철수입니다"  ❌ Faithfulness 낮음 (컨텍스트에 없음)
```

### 2. **Answer Relevancy (답변 관련성)**
- **사용 데이터**: `question`, `answer`
- **평가 방식**:
  - 답변이 질문과 얼마나 관련이 있는지 평가
  - Ground Truth는 **사용하지 않음**
  - 질문에 대한 적절성 평가

**예시:**
```
Question: "캐릭터의 이름은 무엇인가요?"
Answer: "홍길동입니다"  ✅ Relevancy 높음 (질문에 직접 답변)
Answer: "날씨가 좋네요"  ❌ Relevancy 낮음 (질문과 무관)
```

### 3. **Context Precision (컨텍스트 정밀도)**
- **사용 데이터**: `question`, `contexts`, `reference`
- **평가 방식**:
  - 검색된 컨텍스트가 질문과 관련이 있는지 평가
  - `reference` (ground_truth의 첫 번째 값)와 비교
  - 관련 없는 문서가 검색되었는지 확인

**예시:**
```
Question: "캐릭터의 이름은 무엇인가요?"
Reference: "홍길동"
Contexts: [
  "홍길동은 주인공이다",  ✅ 관련 있음
  "이야기는 서울에서 시작된다"  ❌ 관련 없음
]
→ Precision: 0.5 (1개만 관련)
```

### 4. **Context Recall (컨텍스트 재현율)**
- **사용 데이터**: `question`, `contexts`, `ground_truths`
- **평가 방식**:
  - 검색된 컨텍스트가 실제 필요한 정보를 얼마나 포함하는지 평가
  - `ground_truths`와 비교하여 필요한 정보가 검색되었는지 확인
  - **Ground Truth가 필수**

**예시:**
```
Question: "캐릭터의 이름은 무엇인가요?"
Ground Truths: ["홍길동", "길동"]
Contexts: [
  "홍길동은 주인공이다",  ✅ 필요한 정보 포함
  "이야기는 서울에서 시작된다"  ❌ 불필요한 정보
]
→ Recall: 0.5 (필요한 정보의 일부만 검색됨)
```

## 📝 실제 사용 예시

### 평가 데이터셋 작성 예시

```json
[
  {
    "question": "소설의 주인공 이름은 무엇인가요?",
    "ground_truth": "주인공의 이름은 홍길동입니다.",
    "ground_truths": ["홍길동", "길동", "주인공은 홍길동"]
  },
  {
    "question": "주인공의 성격은 어떤가요?",
    "ground_truth": "밝고 활발한 성격입니다.",
    "ground_truths": [
      "밝고 활발함",
      "긍정적이고 에너지가 넘침",
      "사교적이고 친근함"
    ]
  }
]
```

### 평가 실행 과정

```python
# 1. 질문 입력
question = "소설의 주인공 이름은 무엇인가요?"

# 2. RAG 시스템 실행
answer = rag_chain.invoke(question)  
# → "주인공의 이름은 홍길동입니다."

contexts = retriever.get_relevant_documents(question, k=4)
# → ["홍길동은...", "주인공 홍길동이...", ...]

# 3. Ground Truth와 비교
ground_truths = ["홍길동", "길동", "주인공은 홍길동"]

# 4. RAGAS 평가
result = evaluate(
    dataset=Dataset.from_dict({
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
        "ground_truths": [ground_truths],
        "reference": [ground_truths[0]]
    }),
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)
```

## 🎯 Ground Truth 작성 가이드

### 좋은 Ground Truth의 특징:
1. **정확성**: 소설 내용에 기반한 정확한 정보
2. **다양성**: 여러 표현 방식 포함 (ground_truths 배열 사용)
3. **완전성**: 질문에 대한 완전한 답변

### 나쁜 Ground Truth의 예:
```json
{
  "question": "주인공의 이름은?",
  "ground_truth": "이름"  // ❌ 너무 모호함
}
```

### 좋은 Ground Truth의 예:
```json
{
  "question": "주인공의 이름은?",
  "ground_truth": "홍길동",
  "ground_truths": ["홍길동", "주인공은 홍길동", "이름은 홍길동입니다"]  // ✅ 다양한 표현
}
```

## ⚠️ 주의사항

1. **Ground Truth 없이도 평가 가능**: 
   - Faithfulness, Answer Relevancy는 Ground Truth 없이 평가 가능
   - Context Recall은 Ground Truth 필수

2. **Ground Truth 품질이 평가에 영향**:
   - 부정확한 Ground Truth는 잘못된 평가 결과 초래
   - 소설 내용을 정확히 반영해야 함

3. **Questions의 다양성**:
   - 다양한 유형의 질문 포함 (사실 확인, 추론, 성격 묘사 등)
   - 실제 사용자 질문과 유사하게 작성






















