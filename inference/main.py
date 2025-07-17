import json
from rule_loader import load_rules
from rag_retriever import retrieve_rules
from prompt_builder import build_prompt
from qwen_infer import qwen_infer
from query_rewriter import rewrite_query

rules = load_rules([
    './docs/띄어쓰기.json',
    './docs/한글_맞춤법_규정.json',
    './docs/표준어__규정.json',
    './docs/외래어__표기법.json',
    './docs/문장_부호.json'
])

# test 데이터셋 로드
with open('../data/korean_language_rag_V1.0_test.json', encoding='utf-8') as f:
    test_data = json.load(f)

# 10개만 사용
test_data = test_data[:10]

for sample in test_data:
    # Query rewriting 및 sub query 분해
    main_query, sub_queries = rewrite_query(sample)

    # (1) 메인 쿼리 추론
    retrieved_rules = retrieve_rules(main_query, rules)
    prompt = build_prompt(main_query, [], retrieved_rules)
    result = qwen_infer(prompt, model_name='../results')
    print("문항 ID:", sample.get("id", ""))
    print("입력(메인 쿼리):", main_query)
    print("모델 출력:", result)

    # (2) 서브 쿼리별 추론 (선택)
    for idx, sub_q in enumerate(sub_queries):
        sub_rules = retrieve_rules(sub_q, rules)
        sub_prompt = build_prompt(sub_q, [], sub_rules)
        sub_result = qwen_infer(sub_prompt, model_name='../results')
        print(f"서브 쿼리 {idx+1}:", sub_q)
        print(f"서브 쿼리 모델 출력:", sub_result)
    print("="*40)