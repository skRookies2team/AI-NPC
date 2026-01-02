"""
RAGAS를 사용한 RAG 시스템 성능 평가 스크립트

사용 방법:
1. 평가 데이터셋 준비 (evaluation_dataset.json)
2. python evaluate_ragas.py --session_id <세션ID> --dataset evaluation_dataset.json

평가 메트릭:
- faithfulness: 답변이 컨텍스트에 기반한 정확성 (0-1)
- answer_relevancy: 답변이 질문과 관련이 있는지 (0-1)
- context_precision: 검색된 컨텍스트의 정밀도 (0-1)
- context_recall: 검색된 컨텍스트의 재현율 (0-1)
"""

import os
import json
import argparse
from typing import List, Dict
from dotenv import load_dotenv

# RAGAS 관련 임포트
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

# LangChain 관련 임포트
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 환경변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
POSTGRES_CONNECTION_STRING = os.getenv("POSTGRES_CONNECTION_STRING")

if not OPENAI_API_KEY:
    raise ValueError(".env 파일에 OPENAI_API_KEY가 설정되지 않았습니다.")
if not POSTGRES_CONNECTION_STRING:
    raise ValueError(".env 파일에 POSTGRES_CONNECTION_STRING이 설정되지 않았습니다.")


def load_evaluation_dataset(file_path: str) -> List[Dict]:
    """
    평가 데이터셋 로드
    
    데이터셋 형식:
    [
        {
            "question": "질문",
            "ground_truth": "정답 (선택사항)",
            "ground_truths": ["정답1", "정답2"] (선택사항)
        },
        ...
    ]
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def create_rag_chain(session_id: str, character_name: str = "캐릭터", k: int = 6):
    """
    RAG 체인 생성 (ai_server.py와 동일한 구조)
    """
    # PostgreSQL Vector Store에서 retriever 로드
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    collection_name = f"session_{session_id}"
    
    vectorstore = PGVector(
        collection_name=collection_name,
        connection_string=POSTGRES_CONNECTION_STRING,
        embedding_function=embeddings,
        use_jsonb=True
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    
    # 시스템 프롬프트 (범용적, 컨텍스트 기반 답변 강제)
    template = f"""
    당신은 소설 속 인물 '{character_name}'입니다.
    
    **중요 규칙:**
    1. 아래 [Context]에 있는 소설 내용만을 바탕으로 답변하세요.
    2. [Context]에 없는 정보는 절대 만들어내지 마세요.
    3. [Context]에 답변할 수 있는 정보가 없으면 "소설 내용에 그런 정보는 나오지 않습니다" 또는 "모르겠습니다"라고 솔직하게 말하세요.
    4. 컨텍스트 밖의 일반 지식이나 추측을 사용하지 마세요.
    5. 소설에 나오는 인물, 장소, 사건의 이름과 표현을 정확히 사용하세요.
    
    [Context]:
    {{context}}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("human", "{question}")
    ])
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7, openai_api_key=OPENAI_API_KEY)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever


def run_evaluation(
    session_id: str,
    dataset_path: str,
    character_name: str = "캐릭터",
    k: int = 6  # 검색할 문서 개수 (기본값 증가: 4 -> 6)
):
    """
    RAGAS를 사용한 평가 실행
    """
    print(f"📊 RAGAS 평가 시작...")
    print(f"세션 ID: {session_id}")
    print(f"데이터셋: {dataset_path}")
    print("-" * 50)
    
    # 데이터셋 로드
    eval_data = load_evaluation_dataset(dataset_path)
    print(f"✅ 평가 데이터 {len(eval_data)}개 로드 완료")
    
    # RAG 체인 생성
    rag_chain, retriever = create_rag_chain(session_id, character_name, k=k)
    print("✅ RAG 체인 생성 완료")
    
    # 각 질문에 대해 답변 생성 및 컨텍스트 검색
    questions = []
    answers = []
    contexts_list = []
    ground_truths_list = []
    
    print("\n🔄 질문 처리 중...")
    for i, item in enumerate(eval_data):
        question = item["question"]
        questions.append(question)
        
        # Ground truth 처리
        if "ground_truths" in item:
            ground_truths_list.append(item["ground_truths"])
        elif "ground_truth" in item:
            ground_truths_list.append([item["ground_truth"]])
        else:
            ground_truths_list.append([])  # Ground truth가 없으면 빈 리스트
        
        # RAG 체인으로 답변 생성
        print(f"  [{i+1}/{len(eval_data)}] 질문: {question[:50]}...")
        answer = rag_chain.invoke(question)
        answers.append(answer)
        
        # 컨텍스트 검색 (LangChain 최신 버전은 invoke 사용)
        docs = retriever.invoke(question)
        contexts = [doc.page_content for doc in docs]
        contexts_list.append(contexts)
    
    print("\n✅ 모든 질문 처리 완료")
    print("\n📈 RAGAS 메트릭 계산 중...")
    
    # reference 컬럼 생성 (ground_truths의 첫 번째 값 사용)
    # context_precision은 reference가 필요함
    references = []
    for gt_list in ground_truths_list:
        if gt_list and len(gt_list) > 0:
            references.append(gt_list[0])  # 첫 번째 ground truth를 reference로 사용
        else:
            references.append("")  # 빈 문자열
    
    # RAGAS 평가를 위한 데이터셋 생성
    evaluation_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truths": ground_truths_list,
        "reference": references,  # context_precision을 위해 추가
    })
    
    # RAGAS 평가 실행 (embeddings 명시적으로 전달)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
    result = evaluate(
        dataset=evaluation_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,  # embeddings 필요
            context_precision,
            context_recall,
        ],
        embeddings=embeddings,  # RAGAS에 embeddings 전달
    )
    
    # 결과 출력 (EvaluationResult 객체로 접근)
    print("\n" + "=" * 50)
    print("📊 RAGAS 평가 결과")
    print("=" * 50)
    print(f"\n평가된 질문 수: {len(questions)}")
    print(f"\n메트릭 점수:")
    
    # EvaluationResult 객체에서 값 가져오기
    def get_metric_value(metric_name):
        try:
            # 속성으로 접근 시도
            value = getattr(result, metric_name, None)
            if value is None:
                # 딕셔너리처럼 접근 시도
                value = result[metric_name] if hasattr(result, '__getitem__') else None
            if value is None:
                return 0.0
            # 리스트인 경우 평균값 사용
            if isinstance(value, list):
                return sum(value) / len(value) if value else 0.0
            # 숫자인 경우 그대로 사용
            return float(value)
        except (TypeError, ValueError, AttributeError, KeyError):
            return 0.0
    
    faithfulness_score = get_metric_value('faithfulness')
    answer_relevancy_score = get_metric_value('answer_relevancy')
    context_precision_score = get_metric_value('context_precision')
    context_recall_score = get_metric_value('context_recall')
    
    print(f"  - Faithfulness (정확성): {faithfulness_score:.4f}")
    print(f"  - Answer Relevancy (답변 관련성): {answer_relevancy_score:.4f}")
    print(f"  - Context Precision (컨텍스트 정밀도): {context_precision_score:.4f}")
    print(f"  - Context Recall (컨텍스트 재현율): {context_recall_score:.4f}")
    
    # 전체 평균 계산
    avg_score = (faithfulness_score + answer_relevancy_score + context_precision_score + context_recall_score) / 4
    print(f"\n  📌 전체 평균 점수: {avg_score:.4f}")
    
    # 결과를 JSON 파일로 저장
    output_file = f"evaluation_results_{session_id}.json"
    results_dict = {
        "session_id": session_id,
        "num_questions": len(questions),
        "metrics": {
            "faithfulness": faithfulness_score,
            "answer_relevancy": answer_relevancy_score,
            "context_precision": context_precision_score,
            "context_recall": context_recall_score,
            "average": avg_score,
        },
        "detailed_results": result.to_pandas().to_dict("records") if hasattr(result, 'to_pandas') else [],
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 상세 결과가 '{output_file}'에 저장되었습니다.")
    
    return result


def create_sample_dataset(output_path: str = "evaluation_dataset.json"):
    """
    샘플 평가 데이터셋 생성 (예시)
    """
    sample_data = [
        {
            "question": "캐릭터의 이름은 무엇인가요?",
            "ground_truth": "캐릭터의 이름은 [소설에서 나온 이름]입니다.",
        },
        {
            "question": "캐릭터의 성격은 어떤가요?",
            "ground_truth": "캐릭터는 [소설에서 묘사된 성격]입니다.",
        },
        {
            "question": "주요 사건은 무엇이었나요?",
            "ground_truth": "주요 사건은 [소설의 주요 사건]입니다.",
        },
    ]
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 샘플 데이터셋이 '{output_path}'에 생성되었습니다.")
    print("   실제 평가를 위해 질문과 정답을 수정해주세요.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGAS를 사용한 RAG 시스템 평가")
    parser.add_argument(
        "--session_id",
        type=str,
        required=False,
        help="평가할 세션 ID (--create_sample 사용 시 불필요)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="evaluation_dataset.json",
        help="평가 데이터셋 파일 경로 (기본값: evaluation_dataset.json)"
    )
    parser.add_argument(
        "--character_name",
        type=str,
        default="캐릭터",
        help="캐릭터 이름 (기본값: 캐릭터)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=6,
        help="검색할 문서 개수 (기본값: 6)"
    )
    parser.add_argument(
        "--create_sample",
        action="store_true",
        help="샘플 평가 데이터셋 생성"
    )
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_dataset(args.dataset)
    else:
        if not args.session_id:
            print("❌ 오류: --session_id가 필요합니다.")
            print("   샘플 데이터셋을 생성하려면: python evaluate_ragas.py --create_sample")
            exit(1)
        
        if not os.path.exists(args.dataset):
            print(f"❌ 오류: 데이터셋 파일 '{args.dataset}'을 찾을 수 없습니다.")
            print(f"   샘플 데이터셋을 생성하려면 --create_sample 옵션을 사용하세요.")
            exit(1)
        
        run_evaluation(
            session_id=args.session_id,
            dataset_path=args.dataset,
            character_name=args.character_name,
            k=args.k
        )

