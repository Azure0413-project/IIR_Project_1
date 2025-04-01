import json
import torch
from transformers import AutoTokenizer, TextStreamer, AutoModelForCausalLM
from peft import PeftModel

# 設定設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 讀取 JSON 檔案並提取問題
def load_questions(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [item["question"] for item in data]

# 載入 JSON 檔案的問題
question_files = ["./testdata/questions_with_choices_v1.json"]
questions = []
for file in question_files:
    questions.extend(load_questions(file))

# 載入 tokenizer
model_name = "deepseek-ai/deepseek-math-7b-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 載入 **Base Model**
base_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
base_model.eval()

# 載入 **微調後的 LoRA 模型**
fine_tuned_model = PeftModel.from_pretrained(base_model, "./deepseek-math-7b-sft").to(device)
fine_tuned_model.eval()

text_streamer = TextStreamer(tokenizer)

# 生成答案
def generate_answer(model, question):
    prompt = f"{question}\nPlease choose the most appropriate answer from 0, 1, 2, 3 and write the answer in $$, for example, $1$."
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)
    generated_ids = model.generate(**input_ids, max_new_tokens=600)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
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
with open("questions_with_mmlu_choices_v1_0324.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print("結果已儲存到 comparison_results.json")
