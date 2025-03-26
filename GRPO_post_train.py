import wandb
from unsloth import FastLanguageModel, PatchFastRL
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import re
from math_verify import LatexExtractionConfig, parse, verify
from trl import GRPOConfig, GRPOTrainer
PatchFastRL("GRPO", FastLanguageModel)

output_name = "deepseek=math-7B-GRPO_post_train"
dataset_id = "AI-MO/NuminaMath-TIR"
model_id = "deepseek-ai/deepseek-math-7b-base"

#####################################################################################################################
# # wandb.login()
# wandb.login(key="")
# wandb.init(project="", name = )
#####################################################################################################################

##### DATASET #####
train_dataset, test_dataset = load_dataset(dataset_id, split=["train[:5%]", "test[:5%]"])

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

def make_conversation(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
    }

train_dataset = train_dataset.map(make_conversation)
test_dataset = test_dataset.map(make_conversation)
train_dataset = train_dataset.remove_columns(["messages", "problem"])

##### LoRA #####
model, _ = FastLanguageModel.from_pretrained(
    model_name = model_id,
    device_map="auto",
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 8, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = 32,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token 
tokenizer.chat_template = """{% for message in messages %}
{% if message['role'] == 'system' %}System: {{ message['content'] }}
{% elif message['role'] == 'user' %}User: {{ message['content'] }}
{% elif message['role'] == 'assistant' %}Assistant: <think>{{ message['content'].split('<answer>')[0].strip() }}</think><answer>{{ message['content'].split('<answer>')[-1].strip() }}</answer>
{% endif %}{% endfor %}"""

model.print_trainable_parameters()

##### GRPO #####
def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    rewards_list = [1.0 if match else 0.0 for match in matches]
    return [1.0 if match else 0.0 for match in matches]

def accuracy_reward(completions, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    solutions = kwargs["solution"]
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, solution in zip(completion_contents, solutions):
        gold_parsed = parse(solution, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        answer_parsed = parse(content, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        if len(gold_parsed) != 0:
            try:
                rewards.append(float(verify(answer_parsed, gold_parsed)))
            except Exception:
                rewards.append(0.0)
        else:
            rewards.append(1.0)
    return rewards

# Configure training arguments using GRPOConfig
training_args = GRPOConfig(
    learning_rate=1e-5,
    remove_unused_columns=False,  # to access the solution column in accuracy_reward
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    bf16=False,  # 關閉 bf16
    fp16=True,   # 改用 fp16
    
    #####################################################################################################################
    output_dir=output_name,
    # # Parameters that control de data preprocessing
    # max_completion_length=256,  # default: 256
    # num_generations=4,  # default: 8
    # max_prompt_length=512,  # default: 512
    
    # Parameters related to reporting and saving
    report_to="wandb",
    logging_steps=1,
    # push_to_hub=True,
    save_strategy="steps",
    save_steps=10,
    ####################################################################################################################
)

trainer = GRPOTrainer(
    model=model,
    tokenizer = tokenizer,
    reward_funcs=[
        format_reward,
        accuracy_reward
    ],
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset = test_dataset
)

trainer.train()

trainer.save_model(training_args.output_dir)
model.save_pretrained(training_args.output_dir)
