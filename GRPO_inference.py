import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, TextStreamer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "deepseek-ai/deepseek-math-7b-base"
finetune_model_name = "deepseek-math-7b-grpo-lora"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token 

base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    load_in_4bit = True,
    dtype = None,
)

fine_tuned_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = finetune_model_name,
    load_in_4bit = True,
    dtype = None,
)

FastLanguageModel.for_inference(base_model)
FastLanguageModel.for_inference(fine_tuned_model)

text_streamer = TextStreamer(tokenizer)

import json

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

with open("GRPO_v1_comparison_results.json", "w") as f:
    json.dump(results, f, indent=4)
