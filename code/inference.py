import json
import torch
from transformers import AutoTokenizer, TextStreamer
from unsloth import FastLanguageModel

# 設定設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 讀取 JSON 檔案並提取問題
def load_questions(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [item["question"] for item in data]

# 載入兩個 JSON 檔案的問題
question_files = [
    # "./testdata/questions_with_choices_tmmlu_v1.json"
    "./testdata/questions_with_choices_v1.json"
]

questions = []
for file in question_files:
    questions.extend(load_questions(file))

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

# 生成答案，並確保選項格式為 $0$、$1$、$2$、$3$
def generate_answer(model, question):
    # prompt = f"{question}\n請在 A, B, C, D 中選擇一個最適合的答案，並將答案寫在$$之中，例如$A$。"
    prompt = f"{question}\n請在 0, 1, 2, 3 中選擇一個最適合的答案，並將答案寫在$$之中，例如$1$。"
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)
    generated_ids = model.generate(**input_ids, max_new_tokens=300, streamer=text_streamer)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # 確保輸出格式正確
    return generated_text.strip()

# 開始回答
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

# 儲存結果到 JSON
with open("questions_with_choices_v1.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print("結果已儲存到 comparison_results.json")