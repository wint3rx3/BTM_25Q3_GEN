def build_prompt(text, morphs, retrieved_rules):
    rule_text = "\n".join([f"- {r['rule']}" for r in retrieved_rules])
    morph_str = " ".join([f"{m}/{p}" for m, p in morphs])
    prompt = f"""아래 규정과 형태소 분석 결과를 참고하여 입력 문장의 맞춤법, 띄어쓰기, 표준어, 외래어, 문장 부호 오류를 교정하세요.

규정:
{rule_text}

형태소 분석:
{morph_str}

입력 문장:
{text}

수정 결과:"""
    return prompt