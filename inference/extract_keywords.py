from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from tqdm import tqdm
import re

# 시스템 프롬프트 - 키워드 추출용
KEYWORD_SYSTEM_PROMPT = (
    "당신은 한국어 맞춤법·문장부호 교정 답변에서 핵심 키워드를 추출하는 전문가입니다.\n"
    "주어진 답변에서 가장 중요한 어문 규범 관련 키워드를 1~3개 추출하세요.\n"
    "키워드는 단어나 짧은 구문 형태로, 쉼표로 구분하여 제시하세요.\n"
    "예시: 띄어쓰기, 외래어 표기법, 맞춤법"
)

# tokenizer 및 base model 로딩 (train.py와 동일한 방식)
print("🔄 모델 로딩 중...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
model.eval()
print("✅ 모델 로딩 완료!")

# predictions.json 파일 로딩
def load_predictions(file_path):
    """predictions.json 파일을 로딩합니다."""
    try:
        with open(file_path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ JSON 파싱 오류: {e}")
        return None

# answer에서 필요한 부분 추출
def extract_answer_part(answer):
    """answer에서 '옳다' 부분과 이유 부분을 추출합니다."""
    try:
        # '옳다'가 포함된 문장 추출
        pat = r'(?:^|(?<=[.!?…。！？]))\s*(?:(?:"[^"]*"|\'[^\']*\'|[“][^”]*[”]|[‘][^’]*[’]))?[^.!?…。！？]*옳다[^.!?…。！？]*[.!?…。！？]'
        m = re.search(pat, answer)
        if m:
            correct_part = m.group(0)
        else:
            correct_part = ""
        
        # '옳다.' 뒤의 이유 부분 추출
        m = re.search(r'(?<=옳다\.)\s*(.*)$', answer, flags=re.S)
        if m:
            reason = m.group(1)
        else:
            reason = ""
        
        # 추출된 부분을 합쳐서 반환
        extracted_text = (correct_part + " " + reason).strip()
        return extracted_text if extracted_text else answer
    except:
        return answer

# 키워드 추출 프롬프트 생성
def make_keyword_prompt(answer_text):
    """답변 텍스트에서 키워드 추출을 위한 프롬프트를 생성합니다."""
    return f"다음 한국어 맞춤법·문장부호 교정 답변에서 핵심 키워드 1~3개를 추출하세요:\n\n{answer_text}"

# 키워드 추출 함수
def extract_keywords_from_answer(answer_text):
    """단일 답변에서 키워드를 추출합니다."""
    # answer에서 필요한 부분만 추출
    processed_answer = extract_answer_part(answer_text)
    
    messages = [
        {"role": "system", "content": KEYWORD_SYSTEM_PROMPT},
        {"role": "user", "content": make_keyword_prompt(processed_answer)}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True,
        enable_thinking=False  # thinking 모드 비활성화
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=50,  # 키워드는 짧으므로 토큰 수 줄임
            do_sample=True,  # non-thinking 모드에서는 do_sample=True 권장
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,  # non-thinking 모드 권장값
            top_p=0.8,        # non-thinking 모드 권장값
            top_k=20,         # non-thinking 모드 권장값
            min_p=0.0         # non-thinking 모드 권장값
        )

    generated_tokens = output[0][inputs["input_ids"].shape[1]:]
    decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    # 키워드 정제 (쉼표로 분리하고 공백 제거)
    keywords = [kw.strip() for kw in decoded.split(',') if kw.strip()]
    return keywords[:3]  # 최대 3개까지만

# 전체 predictions에서 키워드 추출
def extract_keywords_batch(predictions_data):
    """전체 predictions 데이터에서 키워드를 추출합니다."""
    results = []
    
    for i, item in enumerate(tqdm(predictions_data, desc="키워드 추출 진행")):
        answer_text = item["output"]["answer"]
        keywords = extract_keywords_from_answer(answer_text)
        
        # 모니터링을 위한 출력
        print(f"\n=== 항목 {i+1} (ID: {item['id']}) ===")
        print(f"원본 답변: {answer_text}")
        print(f"추출된 키워드: {', '.join(keywords)}")
        print("-" * 50)
        
        result_item = {
            "id": item["id"],
            "question_type": item["input"]["question_type"],
            "keywords": keywords
        }
        results.append(result_item)
    
    return results

def main():
    """메인 실행 함수"""
    # predictions.json 파일 경로 (현재 디렉토리에서 찾기)
    predictions_files = ["predictions.json", "predictions (1).json"]
    
    predictions_data = None
    used_file = None
    
    for file_path in predictions_files:
        predictions_data = load_predictions(file_path)
        if predictions_data:
            used_file = file_path
            break
    
    if not predictions_data:
        print("❌ predictions.json 파일을 찾을 수 없습니다.")
        return
    
    print(f"📄 사용 파일: {used_file}")
    print(f"📊 총 데이터 수: {len(predictions_data)}")
    
    # 키워드 추출 실행
    print("\n🔄 키워드 추출 시작...")
    keyword_results = extract_keywords_batch(predictions_data)
    
    # 결과 저장
    output_file = "extracted_keywords.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(keyword_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 키워드 추출 완료! 결과 저장: {output_file}")

if __name__ == "__main__":
    main()
