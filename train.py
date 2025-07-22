from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset
import os
import torch
import json

# 0) 공통 상수: system prompt (출력은 id + output.answer 하나만)
SYSTEM_PROMPT = (
    "당신은 한국어 맞춤법·문장부호 교정 전문가입니다.\n"
    "- ‘교정형’: 틀린 부분을 바로잡고, 이유를 어문 규범 근거로 설명하세요.\n"
    "- ‘선택형’: 보기 중 옳은 표현을 선택하고, 이유를 어문 규범 근거로 설명하세요.\n"
    "결과는 JSON으로, 아래 형식만 출력하세요:\n"
    "{\n"
    "  \"id\": \"문항ID\",\n"
    "  \"output\": {\n"
    "    \"answer\": \"정답 문장 및 이유 설명\"\n"
    "  }\n"
    "}\n"
)

# 1) 토크나이저·모델 로드 및 LoRA 설정
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    torch_dtype="auto",
    device_map="auto"
)
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05
)
model = get_peft_model(base_model, lora_config)

# 2) 데이터 로드
# Colab 환경인지 감지
is_colab = "COLAB_GPU" in os.environ

# 데이터 파일 경로
data_files = {
    "train": "data/korean_language_rag_V1.0_train.json",
    "dev":   "data/korean_language_rag_V1.0_dev.json",
    "test":  "data/korean_language_rag_V1.0_test.json"
}

# Colab이면 cache_dir 명시
if is_colab:
    ds = load_dataset("json", data_files=data_files, cache_dir="/content/.cache/hf_datasets")
else:
    ds = load_dataset("json", data_files=data_files)

def make_prompt(item):
    return f"[문항 ID: {item['id']}] 유형: {item['input']['question_type']}\n" + item["input"]["question"]

def preprocess(ex): 
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": make_prompt(ex)}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    target = ex["output"]["answer"]
    tokenized = tokenizer(prompt + target, truncation=True, max_length=4096)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

train_ds = ds["train"].map(preprocess, remove_columns=ds["train"].column_names)
dev_ds   = ds["dev"].map(preprocess,   remove_columns=ds["dev"].column_names)

# 3) Trainer로 학습
training_args = TrainingArguments(
    output_dir="qwen3_correction",
    per_device_train_batch_size=8,                # A100 기준, 8~16까지 가능. 4090에서는 4~8 추천
    gradient_accumulation_steps=4,                # effective batch size 32
    per_device_eval_batch_size=8,
    num_train_epochs=5,                           # 3~10 사이에서 성능 확인
    learning_rate=5e-5,                           # 1e-4 ~ 5e-5 사이에서 튜닝, 5e-5 추천
    fp16=True,                                    # mixed precision (A100/4090 모두 지원)
    logging_steps=10,                             # 더 자주 로그
    save_strategy="epoch",
    evaluation_strategy="epoch",
    predict_with_generate=True,
    warmup_steps=300,                             # 워밍업 증가로 안정적 학습
    weight_decay=0.05,                            # 과적합 방지, 0.01~0.05 추천
    lr_scheduler_type="cosine",                   # 코사인 러닝레이트 스케줄러
    report_to="none",                             # 실전에서는 wandb 등으로 변경 가능
    save_total_limit=3,                           # 저장 모델 개수 제한
    load_best_model_at_end=True,                  # 가장 성능 좋은 모델 자동 로드
    metric_for_best_model="eval_loss",            # dev loss 기준
    dataloader_num_workers=4,                     # 데이터 로딩 속도 향상
)

def collate_fn(batch):
    return {
        "input_ids":      torch.stack([f["input_ids"]      for f in batch]),
        "attention_mask": torch.stack([f["attention_mask"] for f in batch]),
        "labels":         torch.stack([f["labels"]         for f in batch]),
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=dev_ds,
    data_collator=collate_fn
)
trainer.train()
trainer.save_model("qwen3-correction-lora")

# 4) 추론용 모델 로드 (LoRA 포함)
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    torch_dtype="auto",
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "qwen3-correction-lora")
model.eval()

# 5) test 데이터로 일괄 추론
test_ds = ds["test"]  # test엔 output 필드가 없고 id+input만 있음 :contentReference[oaicite:2]{index=2}

def correct_batch(test_dataset):
    results = []
    for ex in test_dataset:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": make_prompt(ex)}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            do_sample=False
        )
        raw = tokenizer.decode(out[0], skip_special_tokens=True)
        pred = json.loads(raw)  # {'id': '750', 'output': {'answer': '…'}}
        # test 구조와 동일하게 input도 붙이려면:
        result = {
            "id":     ex["id"],
            "input":  ex["input"],
            "output": pred["output"]
        }
        results.append(result)
    return results

predictions = correct_batch(test_ds)
# 이제 predictions 요소 하나하나가 train/dev 구조(
# { "id": "...", "input": {...}, "output": { "answer": "..." } }
# )와 정확히 일치합니다.
