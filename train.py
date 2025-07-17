import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

# 모델 및 데이터 경로 설정
model_name = "Qwen/Qwen-7B-Chat"
train_path = "./data/korean_language_rag_V1.0_train.json"
dev_path = "./data/korean_language_rag_V1.0_dev.json"
output_dir = "./results"

# 토크나이저 및 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# 데이터셋 로드
data_files = {"train": train_path, "validation": dev_path}
dataset = load_dataset("json", data_files=data_files)

# 10개만 사용 (train/validation 각각)
dataset["train"] = dataset["train"].select(range(min(10, len(dataset["train"])))) 
dataset["validation"] = dataset["validation"].select(range(min(10, len(dataset["validation"]))))

# 전처리 함수
def preprocess(example):
    prompt = example["input"]["question"]
    answer = example["output"]["answer"]
    # Qwen 스타일 프롬프트
    full_input = f"{prompt}\n정답: "
    full_output = f"{answer}"
    tokenized = tokenizer(
        full_input + full_output,
        truncation=True,
        padding="max_length",
        max_length=512
    )
    tokenized["labels"] = tokenizer(
        full_output,
        truncation=True,
        padding="max_length",
        max_length=512
    )["input_ids"]
    return tokenized

# 데이터셋 토크나이즈
tokenized = dataset.map(preprocess, remove_columns=dataset["train"].column_names)

# 학습 파라미터
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    fp16=True,
    report_to="none"
)

# Trainer 객체 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
)

# 학습 시작
if __name__ == "__main__":
    trainer.train()