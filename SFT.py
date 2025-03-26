from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, get_peft_model

# 1. 加載並統一數據集格式
def unify_dataset(dataset, instruction_field, output_field):
    def format_example(example):
        return {"instruction": example[instruction_field], "output": example[output_field]}
    return dataset.map(format_example, remove_columns=dataset.column_names)

# 加載兩個不同的數據集
dataset1 = load_dataset("Congliu/Chinese-DeepSeek-R1-Distill-data-110k-SFT", split="train")
dataset2 = load_dataset("ServiceNow-AI/R1-Distill-SFT", "v0", split="train")  # 或改成 "v1"

dataset1 = unify_dataset(dataset1, "instruction", "output")
dataset2 = unify_dataset(dataset2, "problem", "reannotated_assistant_content")

# 合併數據集
dataset = concatenate_datasets([dataset1, dataset2])

# 2. 加載模型和 tokenizer
model_name = "deepseek-ai/deepseek-math-7b-base"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 3. 應用 LoRA
lora_config = LoraConfig(
    r=8,  # LoRA rank
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],  # 針對注意力層的 LoRA
)
model = get_peft_model(model, lora_config)

# 4. 格式化數據
def formatting_prompts_func(example):
    return f"### Question: {example['instruction']}\n### Answer: {example['output']}"

response_template = "### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# 5. 設置 Trainer
trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="./fine_tuned_model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=500,
        logging_steps=50,
        learning_rate=2e-4,
        fp16=True,
    ),
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)

# 6. 開始訓練
trainer.train()
