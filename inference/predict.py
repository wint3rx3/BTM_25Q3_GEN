from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import Dataset, Features, Value
import torch
import json
from tqdm import tqdm

# 시스템 프롬프트
SYSTEM_PROMPT = (
    "당신은 한국어 맞춤법·문장부호 교정 전문가입니다.\n"
    "- '교정형': 틀린 부분을 바로잡고, 이유를 어문 규범 근거로 설명하세요.\n"
    "- '선택형': 보기 중 옳은 표현을 선택하고, 이유를 어문 규범 근거로 설명하세요.\n"
    "정답 문장을 자연스럽게 작성하세요."
)

# tokenizer 및 base model 로딩
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", device_map="auto", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base_model, "qwen3-correction-lora")
model.eval()

# 테스트셋 로딩
test_path = "data/korean_language_rag_V1.0_test.json"
with open(test_path, encoding="utf-8") as f:
    test_data = json.load(f)

features = Features({
    "id": Value("string"),
    "input": {
        "question": Value("string"),
        "question_type": Value("string")
    }
})
test_ds = Dataset.from_list(test_data).cast(features)

# 프롬프트 구성 함수
def make_prompt(ex):
    return f"[문항 ID: {ex['id']}] 유형: {ex['input']['question_type']}\n{ex['input']['question']}"

# 추론 함수
def correct_batch(test_dataset):
    results = []
    for ex in tqdm(test_dataset, desc="추론 진행"):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_prompt(ex)}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                temperature=None,  # 명시적으로 None 설정
                top_p=None,        # 명시적으로 None 설정  
                top_k=None         # 명시적으로 None 설정
            )

        generated_tokens = output[0][inputs["input_ids"].shape[1]:]
        decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        results.append({
            "id": ex["id"],
            "input": ex["input"],
            "output": {
                "answer": decoded
            }
        })
    return results

# 결과 생성 및 저장
predictions = correct_batch(test_ds)
with open("predictions.json", "w", encoding="utf-8") as f:
    json.dump(predictions, f, ensure_ascii=False, indent=2)

print("✅ predictions.json 생성 완료!")
