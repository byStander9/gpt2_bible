import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 파인 튜닝된 모델과 토크나이저 로드
model_name = "./fine_tuned_gpt2"  # 모델이 저장된 경로
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# GPU 사용 가능하면 GPU 사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 모델을 평가 모드로 전환 (학습 모드에서 평가 모드로)
model.eval()

# 입력 텍스트
prompt = "예수님께서 십자가에 못박힌 부분 알려줘."

# 프롬프트를 토큰화
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# 모델에서 응답 생성
with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        max_length=1000,  # 응답의 최대 길이 설정
        num_return_sequences=1,  # 생성할 응답의 수
        temperature=0.7,  # 텍스트 생성의 다양성 조절
        top_p=0.9,  # 확률 분포에서 상위 p%를 선택
        do_sample=True  # 확률적으로 응답 생성
    )

# 응답 디코딩
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 결과 출력
print("응답:", response)
