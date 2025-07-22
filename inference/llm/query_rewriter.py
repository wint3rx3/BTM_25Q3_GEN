from inference.llm.qwen_infer import qwen_infer
from inference.analysis.morph_analyzer import analyze_morphs

def rewrite_query(sample):
    """
    LLM을 이용해 임베딩 검색에 적합한 정보 중심 쿼리 및 서브 쿼리 생성
    반환: (main_query, [sub_query 리스트])
    """
    import re
    question = sample["input"]["question"]
    qtype = sample["input"].get("question_type", "")

    # 따옴표 안의 문장 추출
    quoted = re.findall(r'\"([^\"]+)\"', question)
    if quoted:
        main_sentence = quoted[0]
    else:
        main_sentence = question

    # 형태소 분석
    morphs = analyze_morphs(main_sentence)

    # LLM 프롬프트 구성
    prompt = (
        "아래는 입력 문장과 형태소 분석 결과입니다.\n"
        "이 정보를 바탕으로, 어문 규범 json의 규정 설명문과 유사한 정보 중심 쿼리(짧은 문장)로 변환해줘.\n"
        f"입력 문장: {main_sentence}\n"
        f"형태소 분석: {morphs}\n"
        f"질문 유형: {qtype}\n"
        "출력 예시: '녀자 두음법칙', '오뚝한 표준어 여부', '학교에갔다 띄어쓰기 오류' 등"
    )

    # LLM으로 정보 중심 쿼리 생성
    main_query = qwen_infer(prompt).strip().split('\n')[0]  # 첫 줄만 사용

    # (선택) 서브 쿼리도 LLM으로 생성하고 싶으면 아래처럼 추가
    sub_queries = []
    if qtype == "교정형":
        sub_prompt = (
            "아래는 입력 문장과 형태소 분석 결과입니다.\n"
            "이 정보를 바탕으로, 어문 규범 오류 유형별(맞춤법, 띄어쓰기, 표준어, 외래어, 문장 부호) 정보 중심 쿼리(짧은 문장) 리스트를 만들어줘.\n"
            f"입력 문장: {main_sentence}\n"
            f"형태소 분석: {morphs}\n"
            "출력 예시: ['학교에갔다 맞춤법 오류', '학교에갔다 띄어쓰기 오류', ...]"
        )
        llm_sub = qwen_infer(sub_prompt)
        # 리스트 형태로 파싱 (예: "['...', '...']" 또는 줄바꿈 구분)
        import ast
        try:
            sub_queries = ast.literal_eval(llm_sub)
        except Exception:
            sub_queries = [line.strip() for line in llm_sub.split('\n') if line.strip()]
    elif qtype == "선택형":
        # 선택형은 선택지별로 쿼리 생성
        choices = re.findall(r"\{(.+?)\}", question)
        if choices:
            for choice in choices[0].split('/'):
                choice_prompt = (
                    "아래는 선택지와 형태소 분석 결과입니다.\n"
                    "이 정보를 바탕으로, 어문 규범 json의 규정 설명문과 유사한 정보 중심 쿼리(짧은 문장)로 변환해줘.\n"
                    f"선택지: {choice.strip()}\n"
                    f"입력 문장: {main_sentence}\n"
                    f"형태소 분석: {morphs}\n"
                    "출력 예시: '불을 표기 어문 규범', '불 표기 어문 규범'"
                )
                sub_query = qwen_infer(choice_prompt).strip().split('\n')[0]
                sub_queries.append(sub_query)

    return main_query, sub_queries