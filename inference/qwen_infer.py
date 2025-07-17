from transformers import AutoTokenizer, AutoModelForCausalLM

def qwen_infer(prompt, model_name='naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B'):
    # 모델 및 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    
    # chat 템플릿 구성
    chat = [
        {"role": "tool_list", "content": ""},
        {"role": "system", "content": "You are CLOVA X, an AI language model created by NAVER. Today is April 24, 2025 (Thursday)."},
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    inputs = inputs.to("cuda")
    output_ids = model.generate(
        **inputs,
        max_length=1024,
        stop_strings=["<|endofturn|>", "<|stop|>"],
        tokenizer=tokenizer
    )
    return tokenizer.batch_decode(output_ids)[0]