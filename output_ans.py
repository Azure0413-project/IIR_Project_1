import json

def json_to_text(input_json_file, answer_json_file, output_text_file):
    # 讀取 LLM 回答的 JSON 檔案
    with open(input_json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # 讀取 正確答案 的 JSON 檔案
    with open(answer_json_file, 'r', encoding='utf-8') as f:
        answer_data = json.load(f)
    
    """
    # 確保兩個 JSON 檔案的順序一致
    assert len(json_data) == len(answer_data), "兩個 JSON 檔案的題目數量不一致"
    """

    # 寫入文字檔案
    with open(output_text_file, 'w', encoding='utf-8') as f:
        for i, (item, ans) in enumerate(zip(json_data, answer_data), start=1):
            f.write(f"### 題目 {i} (正解: {ans['answer']})\n")
            f.write("**Question:**\n")
            f.write(item["question"] + "\n\n")
            f.write("**Base Model 回答:**\n")
            f.write(item["base_answer"] + "\n\n")
            f.write("---\n\n")

# 指定輸入 JSON 檔案和輸出文本檔案
#input_json_file = "./questions_with_choices_v1.json"  # 請替換為你的 JSON 檔案名稱
input_json_file = "./questions_with_tmmlu_choices_v1_0324.json"
answer_json_file = "./questions_with_choices_tmmlu_v1.json"
output_text_file = "tmmlu_0324_output.txt"

# 執行轉換
# 執行轉換
json_to_text(input_json_file, answer_json_file, output_text_file)
print("轉換完成，結果已儲存至 output.txt")