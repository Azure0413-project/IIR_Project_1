import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
from datasets import load_dataset
from tqdm import tqdm
from unsloth import FastLanguageModel

# 設定設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用設備: {device}")

# 載入 MMLU-Pro-CoT-Train-Labeled 資料集
try:
    dataset = load_dataset("UW-Madison-Lee-Lab/MMLU-Pro-CoT-Train-Labeled")
    # 如果是由 datasets 載入，通常會有 'train' 分割
    if 'train' in dataset:
        working_dataset = dataset['train']
    else:
        working_dataset = dataset
    print(f"成功載入資料集，資料筆數: {len(working_dataset)}")
except Exception as e:
    print(f"從 Hugging Face 載入失敗: {e}")
    print("請確保資料集路徑正確或手動下載資料集")
    exit(1)

# 檢查資料集結構
sample_item = working_dataset[0] if len(working_dataset) > 0 else {}
print(f"資料集欄位: {sample_item.keys()}")

# 判斷問題和鏈思考的欄位名稱
question_field = "question"  # 根據你的描述，問題欄位是 "question"
cot_field = "chain_of_thoughts"  # 假設 CoT 欄位是 "chain_of_thoughts"

# 確認欄位存在
if question_field not in sample_item:
    print(f"警告: 找不到問題欄位 '{question_field}'")
    question_field = input("請輸入正確的問題欄位名稱: ")

if cot_field not in sample_item:
    print(f"警告: 找不到思考鏈欄位 '{cot_field}'")
    cot_field = input("請輸入正確的思考鏈欄位名稱: ")


# 載入 tokenizer
model_name = "deepseek-ai/deepseek-math-7b-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
# 載入 base model
print(f"載入基礎模型: {model_name}")
# 載入 **微調前** 模型 (Base)
base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    dtype=None,
    load_in_4bit=True
)
# 載入微調後的 LoRA 模型
print("載入微調後的 LoRA 模型")
# 載入 **微調後** 模型 (Fine-tuned)
fine_tuned_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./deepseek-math-7b-orpo-lora",  # 你的微調後模型存放路徑
    dtype=None,
    load_in_4bit=True
)

# 生成答案
def generate_answer(model, question):
    # 使用簡單提示格式，可以根據實際需求調整
    prompt = question
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **input_ids, 
            max_new_tokens=200,
            do_sample=False
        )
    
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # 移除原始問題，只保留生成的答案部分
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()
    
    return generated_text

# 創建 JSONL 文件
results = []
output_file = "mmlu_pro_comparison.jsonl"

# 處理資料集中的問題
print("開始處理問題並生成回答...")
for i, item in enumerate(tqdm(working_dataset)):
    question = item[question_field]
    chain_of_thoughts = item[cot_field]
    
    # 使用微調的模型生成答案
    try:
        fine_tuned_answer = generate_answer(fine_tuned_model, question)
        
        # 加入結果
        results.append({
            "prompt": question,
            "chosen": chain_of_thoughts,
            "reject": fine_tuned_answer
        })
        
        # 每處理 20 個問題保存一次結果，避免中斷時丟失資料
        if (i + 1) % 20 == 0:
            with open(output_file, "w", encoding="utf-8") as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"處理問題 {i+1} 時出錯: {e}")

# 最終保存結果
with open(output_file, "w", encoding="utf-8") as f:
    for result in results:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

print(f"所有結果已儲存到 {output_file}，共 {len(results)} 筆資料")
