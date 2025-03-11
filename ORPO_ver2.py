import torch
import json
from datasets import load_dataset
from trl import ORPOConfig, ORPOTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import AutoTokenizer, AutoModelForCausalLM

# **函式: 格式化 Prompt**
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a expert with advanced knowledge in speaking chinese and english. 
Please answer the following medical question. 

### Question:
{}

### Response:
<think>
{}
</think>
{}"""

def format_prompt(sample, EOS_TOKEN):
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
    instruction = "You are an AI assistant. You will be given a task. You must generate a detailed and correct answer. And you should also know the knowledge in chinese."
    input = sample["question"]
    accepted = tokenizer.apply_chat_template(sample["chosen"], tokenize=False)
    rejected = tokenizer.apply_chat_template(sample["rejected"], tokenize=False)
    sample["prompt"] = alpaca_prompt.format(instruction, input, "")
    sample["chosen"] = accepted + EOS_TOKEN
    sample["rejected"] = rejected + EOS_TOKEN
    return sample

# **函式: 寫入 JSON**
def write_json(output_data, output_file):
    """將模型輸出寫入 JSON"""
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)

# **設置模型名稱**
model_name = "deepseek-ai/deepseek-math-7b-base"

# **確認 GPU 和數據類型**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16
print(f"Using {device} device\n")

# **載入 Tokenizer**
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token 

max_seq_length = 2048 
dtype = None 
load_in_4bit = True

# **載入模型**
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# **載入數據集**
dataset = load_dataset("mlabonne/orpo-dpo-mix-40k", split="train")

EOS_TOKEN = tokenizer.eos_token
dataset = dataset.map(format_prompt, fn_kwargs={"EOS_TOKEN": EOS_TOKEN}, batched=False)
dataset = dataset.train_test_split(test_size=0.01)

# **微調 LoRA**
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=64,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407
)

import wandb
wandb.login(key="0fc9b0afa729d0aff22088c9788ccf473893d242")  # 替換成你的 API Key

training_args = ORPOConfig(
    per_device_train_batch_size=2,  # 根據 GPU 記憶體調整
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,  # 累積梯度來適應更小的 batch size
    beta=0.1,  # ORPO Beta 參數，控制約束
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    max_steps=500,  # 訓練步數
    num_train_epochs=1,
    optim="adamw_torch",
    weight_decay=0.01,
    max_grad_norm=1.0,
    warmup_ratio=0.03,
    max_length=1024,
    max_prompt_length=512,
    max_completion_length=512,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_strategy="steps",
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=1,
    output_dir="./deepseek-math-7b-orpo-lora-big",
    save_strategy="epoch",
    report_to="wandb"
)


orpo_trainer = ORPOTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    args=training_args,
    processing_class=tokenizer
)

orpo_trainer.train()

# **儲存微調後的模型**
model.save_pretrained("./deepseek-math-7b-orpo-lora-big")