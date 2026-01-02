"""
RAGAS 평가를 실행하고 결과를 출력하는 스크립트
실제 세션이 없어도 샘플 데이터로 평가를 실행할 수 있습니다.
"""

import os
import json
import sys
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# RAGAS와 필요한 라이브러리 임포트
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from datasets import Dataset
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_community.vectorstores import PGVector
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
except ImportError as e:
    print(f"❌ 필요한 패키지가 설치되지 않았습니다: {e}")
    print("다음 명령어로 설치해주세요: pip install ragas datasets")
    sys.exit(1)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
POSTGRES_CONNECTION_STRING = os.getenv("POSTGRES_CONNECTION_STRING")

if not OPENAI_API_KEY:
    print("❌ OPENAI_API_KEY가 설정되지 않았습니다.")
    sys.exit(1)


def create_sample_evaluation_data():
    """샘플 평가 데이터 생성"""
    return [
        {
            "question": "캐릭터의 이름은 무엇인가요?",
            "ground_truth": "캐릭터의 이름은 소설에서 확인할 수 있습니다.",
        },
        {
            "question": "캐릭터의 성격은 어떤가요?",
            "ground_truth": "캐릭터의 성격은 소설 내용에 기반합니다.",
        },
        {
            "question": "주요 사건은 무엇이었나요?",
            "ground_truth": "주요 사건은 소설의 내용에 포함되어 있습니다.",
        },
    ]


def run_evaluation_with_session(session_id: str, character_name: str = "캐릭터"):
    """실제 세션으로 평가 실행"""
    print(f"📊 RAGAS 평가 시작...")
    print(f"세션 ID: {session_id}")
    print("-" * 50)
    
    try:
        # PostgreSQL Vector Store에서 retriever 로드
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        collection_name = f"session_{session_id}"
        
        vectorstore = PGVector(
            collection_name=collection_name,
            connection_string=POSTGRES_CONNECTION_STRING,
            embedding_function=embeddings,
            use_jsonb=True
        )
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
        print("✅ 벡터 스토어 로드 완료")
        
    except Exception as e:
        print(f"❌ 벡터 스토어 로드 실패: {e}")
        print("샘플 데이터로 평가를 진행합니다...")
        return run_sample_evaluation()
    
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
    
    # 평가 데이터 준비
    eval_data = create_sample_evaluation_data()
    print(f"✅ 평가 데이터 {len(eval_data)}개 준비 완료")
    
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
            ground_truths_list.append([])
        
        # RAG 체인으로 답변 생성
        print(f"  [{i+1}/{len(eval_data)}] 질문: {question[:50]}...")
        try:
            answer = rag_chain.invoke(question)
            answers.append(answer)
            
            # 컨텍스트 검색 (LangChain 최신 버전은 invoke 사용)
            docs = retriever.invoke(question)
            contexts = [doc.page_content for doc in docs[:4]]  # 4개만 선택
            contexts_list.append(contexts)
        except Exception as e:
            print(f"    ⚠️ 오류 발생: {e}")
            answers.append("답변 생성 실패")
            contexts_list.append([])
    
    return evaluate_metrics(questions, answers, contexts_list, ground_truths_list, session_id)


def run_sample_evaluation():
    """샘플 데이터로 평가 실행 (실제 세션 없이)"""
    print("📊 RAGAS 샘플 평가 시작...")
    print("(실제 세션이 없어 샘플 데이터로 평가합니다)")
    print("-" * 50)
    
    # 샘플 데이터 생성
    questions = [
        "캐릭터의 이름은 무엇인가요?",
        "캐릭터의 성격은 어떤가요?",
        "주요 사건은 무엇이었나요?",
    ]
    
    # 샘플 답변 (실제로는 RAG 체인에서 생성됨)
    answers = [
        "캐릭터의 이름은 소설 내용에서 확인할 수 있습니다.",
        "캐릭터는 소설에 묘사된 성격을 가지고 있습니다.",
        "주요 사건은 소설의 줄거리에 포함되어 있습니다.",
    ]
    
    # 샘플 컨텍스트
    contexts_list = [
        ["소설의 첫 번째 부분입니다. 캐릭터에 대한 설명이 포함되어 있습니다."],
        ["소설의 두 번째 부분입니다. 캐릭터의 성격이 묘사되어 있습니다."],
        ["소설의 세 번째 부분입니다. 주요 사건이 전개됩니다."],
    ]
    
    ground_truths_list = [
        ["캐릭터의 이름은 소설에서 확인할 수 있습니다."],
        ["캐릭터의 성격은 소설 내용에 기반합니다."],
        ["주요 사건은 소설의 내용에 포함되어 있습니다."],
    ]
    
    print(f"✅ 샘플 데이터 {len(questions)}개 준비 완료")
    
    return evaluate_metrics(questions, answers, contexts_list, ground_truths_list, "sample")


def evaluate_metrics(questions, answers, contexts_list, ground_truths_list, session_id):
    """RAGAS 메트릭 평가"""
    print("\n📈 RAGAS 메트릭 계산 중...")
    
    try:
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
        
        # 결과 출력
        print_results(result, len(questions), session_id)
        
        return result
        
    except Exception as e:
        print(f"❌ 평가 실행 중 오류 발생: {e}")
        print("\n💡 참고: 실제 평가를 위해서는:")
        print("   1. PostgreSQL에 벡터 스토어가 있어야 합니다")
        print("   2. OPENAI_API_KEY가 설정되어 있어야 합니다")
        print("   3. ragas와 datasets 패키지가 설치되어 있어야 합니다")
        return None


def print_results(result, num_questions, session_id):
    """결과 출력"""
    print("\n" + "=" * 60)
    print("📊 RAGAS 평가 결과")
    print("=" * 60)
    print(f"\n평가된 질문 수: {num_questions}")
    print(f"\n📈 메트릭 점수:")
    
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
    
    metrics = {
        "faithfulness": get_metric_value('faithfulness'),
        "answer_relevancy": get_metric_value('answer_relevancy'),
        "context_precision": get_metric_value('context_precision'),
        "context_recall": get_metric_value('context_recall'),
    }
    
    for metric_name, score in metrics.items():
        score_value = float(score) if hasattr(score, '__float__') else 0.0
        bar_length = int(score_value * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        
        metric_kr = {
            "faithfulness": "Faithfulness (정확성)",
            "answer_relevancy": "Answer Relevancy (답변 관련성)",
            "context_precision": "Context Precision (컨텍스트 정밀도)",
            "context_recall": "Context Recall (컨텍스트 재현율)",
        }.get(metric_name, metric_name)
        
        print(f"  {metric_kr:35s}: {score_value:.4f} [{bar}]")
    
    # 전체 평균 계산
    avg_score = sum(metrics.values()) / len(metrics) if metrics else 0
    avg_bar = "█" * int(avg_score * 40) + "░" * (40 - int(avg_score * 40))
    print(f"\n  {'전체 평균 점수':35s}: {avg_score:.4f} [{avg_bar}]")
    
    # 결과 해석
    print("\n📝 결과 해석:")
    if avg_score >= 0.8:
        print("  ✅ 우수한 성능입니다!")
    elif avg_score >= 0.6:
        print("  ⚠️ 보통 성능입니다. 개선 여지가 있습니다.")
    else:
        print("  ❌ 성능이 낮습니다. 시스템 개선이 필요합니다.")
    
    # 결과 저장
    output_file = f"evaluation_results_{session_id}.json"
    results_dict = {
        "session_id": session_id,
        "num_questions": num_questions,
        "metrics": {k: float(v) if hasattr(v, '__float__') else 0.0 for k, v in metrics.items()},
        "average": float(avg_score),
    }
    
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)
        print(f"\n💾 결과가 '{output_file}'에 저장되었습니다.")
    except Exception as e:
        print(f"\n⚠️ 결과 저장 실패: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAGAS 평가 실행")
    parser.add_argument(
        "--session_id",
        type=str,
        help="평가할 세션 ID (없으면 샘플 평가 실행)"
    )
    parser.add_argument(
        "--character_name",
        type=str,
        default="캐릭터",
        help="캐릭터 이름"
    )
    
    args = parser.parse_args()
    
    if args.session_id:
        run_evaluation_with_session(args.session_id, args.character_name)
    else:
        print("세션 ID가 제공되지 않아 샘플 평가를 실행합니다.")
        print("실제 세션으로 평가하려면: python run_evaluation.py --session_id <세션ID>\n")
        run_sample_evaluation()

