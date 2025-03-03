import torch
from transformers import AutoTokenizer, TextStreamer
from unsloth import FastLanguageModel

# 設定設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 載入 tokenizer
model_name = "deepseek-ai/deepseek-math-7b-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 載入 **微調前** 模型 (Base)
base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    dtype=None,
    load_in_4bit=True
)

# 載入 **微調後** 模型 (Fine-tuned)
fine_tuned_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./deepseek-math-7b-orpo-lora",  # 你的微調後模型存放路徑
    dtype=None,
    load_in_4bit=True
)

# 讓兩個模型進入推理模式
FastLanguageModel.for_inference(base_model)
FastLanguageModel.for_inference(fine_tuned_model)

text_streamer = TextStreamer(tokenizer)

questions = [
    "Solve the equation: x^2 - 4x + 4 = 0",
    "What is the derivative of sin(x)?",
    "Integrate 3x^2 + 2x + 1 with respect to x.",
    "Find the determinant of the matrix [[1,2],[3,4]].",
    "What is the Taylor series expansion of e^x?",
]

def generate_answer(model, question):
    input_ids = tokenizer(question, return_tensors="pt").to(device)
    generated_ids = model.generate(**input_ids, max_new_tokens=100, streamer=text_streamer)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

results = []

for question in questions:
    print("\n===============================")
    print(f"Question: {question}")

    # **Base Model 回答**
    print("\n[Base Model Answer]")
    base_answer = generate_answer(base_model, question)
    print(base_answer)

    # **Fine-Tuned Model 回答**
    print("\n[Fine-Tuned Model Answer]")
    fine_tuned_answer = generate_answer(fine_tuned_model, question)
    print(fine_tuned_answer)

    results.append({
        "question": question,
        "base_answer": base_answer,
        "fine_tuned_answer": fine_tuned_answer
    })

import json

with open("comparison_results.json", "w") as f:
    json.dump(results, f, indent=4)