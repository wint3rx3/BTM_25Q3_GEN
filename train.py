from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import Trainer as HfTrainer
from datasets import Dataset, Features, Value
import os
import torch
import json

# 시스템 프롬프트 (자연어 출력 유도)
SYSTEM_PROMPT = (
    "당신은 한국어 맞춤법·문장부호 교정 전문가입니다.\n"
    "- '교정형': 틀린 부분을 바로잡고, 이유를 어문 규범 근거로 설명하세요.\n"
    "- '선택형': 보기 중 옳은 표현을 선택하고, 이유를 어문 규범 근거로 설명하세요.\n"
    "정답 문장을 자연스럽게 작성하세요."
)

# 1) 모델 및 LoRA 설정
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,                   # 8-bit 로딩
    llm_int8_threshold=6.0,              # 엔코더/디코더 스케일 기준값
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B", 
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True
)

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05
)
model = get_peft_model(base_model, lora_config)

# 필수 설정: gradient checkpointing & input grads 활성화
model.config.use_cache = False
model.gradient_checkpointing_enable()    # 메모리 절약을 위한 gradient checkpointing
model.enable_input_require_grads()       # 8-bit 모델의 gradient 흐름 활성화

# LoRA adapter만 학습 가능하도록 설정 (LoRA가 아닌 파라미터만 freeze)
for name, param in model.named_parameters():
    if "lora" not in name:
        param.requires_grad = False

# 학습 가능한 파라미터 확인
model.print_trainable_parameters()

# 커스텀 Trainer 정의
class Trainer(HfTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # inputs 에는 'input_ids','attention_mask','labels'가 포함되어 있음
        labels = inputs.get("labels")
        # PeftModelForCausalLM은 **inputs 로 labels를 넘기면 loss를 리턴
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

# 2) 데이터셋 로드
data_files = {
    "train": "data/korean_language_rag_V1.0_train.json",
    "dev":   "data/korean_language_rag_V1.0_dev.json",
    "test":  "data/korean_language_rag_V1.0_test.json"
}

# 데이터셋 스키마 정의
input_struct = {
    "question":      Value("string"),
    "question_type": Value("string"),
}
features_with_labels = Features({
    "id":    Value("string"),
    "input": input_struct,
    "output": {
        "answer": Value("string")
    }
})
features_no_labels = Features({
    "id":    Value("string"),
    "input": input_struct,
})

# JSON 파일 직접 읽어 Dataset.from_list로 변환
def load_json_dataset(path, features):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data).cast(features)

train_ds = load_json_dataset(data_files["train"], features_with_labels)
dev_ds   = load_json_dataset(data_files["dev"],   features_with_labels)
test_ds  = load_json_dataset(data_files["test"],  features_no_labels)

# 3) 전처리
def make_prompt(item):
    return f"[문항 ID: {item['id']}] 유형: {item['input']['question_type']}\n{item['input']['question']}"

def preprocess(ex):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": make_prompt(ex)}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    target = ex["output"]["answer"]
    full = prompt + target

    tokenized = tokenizer(
        full,
        truncation=True,
        max_length=1024,                 # 메모리 절약을 위해 2048에서 1024로 감소
        padding="max_length"
    )

    prompt_len = len(tokenizer(prompt, truncation=True, max_length=1024)["input_ids"])
    labels = [-100] * prompt_len + tokenized["input_ids"][prompt_len:]
    tokenized["labels"] = labels[:1024]  # ensure max_length

    return tokenized

train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names)
dev_ds   = dev_ds.map(preprocess,   remove_columns=dev_ds.column_names)

# PyTorch Tensor 로딩을 위한 포맷 설정
train_ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])
dev_ds.set_format(type="torch",   columns=["input_ids","attention_mask","labels"])

# 4) Trainer 설정
training_args = TrainingArguments(
    output_dir="qwen3_correction",
    per_device_train_batch_size=1,         # 메모리 절약을 위해 배치 크기 최소화
    gradient_checkpointing=True,           # 메모리 절감
    bf16=True,                             # A100이면 추천 (fp16보다 안정)
    fp16=False,                            # fp16 대신 bf16 사용
    # deepspeed="deepspeed_config.json",   # DeepSpeed ZeRO Stage 2 (필요시 주석 해제)
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    report_to="none"                       # wandb 비활성화
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=dev_ds,
    # tokenizer=tokenizer 제거 (deprecated in transformers 5.0.0)
)

trainer.train()
trainer.save_model("qwen3-correction-lora")

# 5) 테스트셋 추론
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(base_model, "qwen3-correction-lora")
model.eval()

def correct_batch(test_dataset):
    results = []
    for ex in test_dataset:
        # 프롬프트 구성
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_prompt(ex)}
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 입력 토큰화
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 텍스트 생성
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
                # temperature=0.0 제거 (do_sample=False일 때 불필요)
            )
        
        # 디코딩 (프롬프트 부분 제거)
        generated_tokens = output[0][inputs['input_ids'].shape[1]:]
        decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # 결과 구성
        result = {
            "id": ex["id"],
            "input": ex["input"],
            "output": {
                "answer": decoded
            }
        }
        results.append(result)
    
    return results

predictions = correct_batch(test_ds)

# 결과 저장
with open("predictions.json", "w", encoding="utf-8") as f:
    json.dump(predictions, f, ensure_ascii=False, indent=2)
