from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset, Features, Value
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

# 2) 데이터 로드: train/dev와 test에 서로 다른 스키마 지정
is_colab = "COLAB_GPU" in os.environ
data_files = {
    "train": "data/korean_language_rag_V1.0_train.json",
    "dev":   "data/korean_language_rag_V1.0_dev.json",
    "test":  "data/korean_language_rag_V1.0_test.json"
}
cache_dir = "/content/.cache/hf_datasets" if is_colab else None

# ── 공통 input struct 정의
input_struct = {
    "question":      Value("string"),
    "question_type": Value("string"),
}

# ── train/dev용: output.answer 포함
features_with_labels = Features({
    "id":    Value("string"),
    "input": input_struct,
    "output": {
        "answer": Value("string")
    }
})
train_ds = load_dataset(
    "json",
    data_files={"train": data_files["train"]},
    features=features_with_labels,
    cache_dir=cache_dir
)["train"]
dev_ds   = load_dataset(
    "json",
    data_files={"dev": data_files["dev"]},
    features=features_with_labels,
    cache_dir=cache_dir
)["dev"]

# ── test용: output 필드 없음
features_no_labels = Features({
    "id":    Value("string"),
    "input": input_struct,
})
test_ds = load_dataset(
    "json",
    data_files={"test": data_files["test"]},
    features=features_no_labels,
    cache_dir=cache_dir
)["test"]

# 3) 전처리 함수
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

train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names)
dev_ds   = dev_ds.map(preprocess,   remove_columns=dev_ds.column_names)

# 4) Trainer로 학습
training_args = TrainingArguments(
    output_dir="qwen3_correction",       # 체크포인트가 저장될 디렉터리
    per_device_train_batch_size=8,       # GPU 1장 기준 배치 사이즈
    num_train_epochs=3,                  # 데이터 크기에 맞춰 epoch 수 설정
    evaluation_strategy="epoch",         # 매 epoch 끝날 때마다 평가
    save_strategy="epoch",               # 매 epoch 끝날 때마다 모델 저장
    logging_steps=50,                    # 50 스텝마다 로그 출력
    fp16=True                            # 가능하면 mixed-precision 사용
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

# 5) 추론용 모델 로드 및 test 데이터 처리
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    torch_dtype="auto",
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "qwen3-correction-lora")
model.eval()

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
        pred = json.loads(raw)
        result = {
            "id":     ex["id"],
            "input":  ex["input"],
            "output": pred["output"]
        }
        results.append(result)
    return results

predictions = correct_batch(test_ds)
# predictions를 원하는 형식으로 저장/후처리
