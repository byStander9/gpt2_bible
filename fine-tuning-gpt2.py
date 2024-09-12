import os
import json
import pandas as pd
import torch
from datasets import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer


def load_bible_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


file_path = 'C:/Users/byStander/Documents/bible_toyProject/bible_data.json'
bible_data = load_bible_data(file_path)

# 데이터를 Hugging Face Datasets 포맷으로 변환
data = []
for entry in bible_data:
    text = entry["text"]
    reference = f'{entry["book"]} {entry["chapter"]}:{entry["verse"]}'
    combined_text = f'{text} ({reference})'
    data.append({"input_text": combined_text})


dataset = Dataset.from_pandas(pd.DataFrame(data))

# GPT-2 모델과 토크나이저 로드
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# GPT-2 모델은 패딩 토큰이 없으므로 패딩 토큰을 추가
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id


def tokenize_function(examples):
    tokenized_inputs = tokenizer(examples["input_text"], padding="max_length", truncation=True, max_length=512)
    # 라벨을 input_ids와 동일하게 설정
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs

# 데이터셋을 토큰화합니다.
tokenized_dataset = dataset.map(tokenize_function, batched=True)

print(f"데이터셋 크기: {len(tokenized_dataset)}")

# 학습 arguments 추가
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="no",
    save_steps=100,
    save_total_limit=2,
    learning_rate=5e-5,  # 학습률 조정
    lr_scheduler_type="linear",  # 선형 학습률 스케줄러 사용
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    fp16=False,  # FP16 비활성화
    report_to="none",
    logging_first_step=True,
    max_grad_norm=1.0  # Gradient Clipping 활성화
)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.get("loss")
        # print(f"Loss during training: {loss}")  # 손실 값 출력
        return (loss, outputs) if return_outputs else loss

# NaN 및 Inf 값 감지 함수
def detect_anomalies():
    for param in model.parameters():
        if torch.isnan(param).any():
            print("NaN detected in model parameters")
        if torch.isinf(param).any():
            print("Infinity detected in model parameters")

# CustomTrainer 설정
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

trainer.train()

detect_anomalies()

model.save_pretrained("./fine_tuned_gpt2")
tokenizer.save_pretrained("./fine_tuned_gpt2")
