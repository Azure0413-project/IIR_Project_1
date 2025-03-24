import wandb
################################################ ADJUST ##########################################################
# wandb.login()
wandb.login(key="")
wandb.init(project="", name="",)
output_dir="Qwen2-0.5B-GRPO-post-train_orca_dpo"
######################################################################################################################
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

from datasets import load_dataset, Dataset
import torch, re
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from math_verify import LatexExtractionConfig, parse, verify

# dataset_id = "AI-MO/NuminaMath-TIR"
# train_dataset, test_dataset = load_dataset(dataset_id, split=["train[:5%]", "test[:5%]"])
dataset_id = "Intel/orca_dpo_pairs"
dataset = load_dataset(dataset_id, split=["train"])
train_dataset, test_dataset = load_dataset(dataset_id, split=["train[:60%]", "train[40%:]"])
# print(train_dataset)
# print(train_dataset[0])

def make_conversation(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            # {"role": "user", "content": example["problem"]}, # AI-MO/NuminaMath-TIR
            {"role": "user", "content": example["question"]}, # Intel/orca_dpo_pairs
        ],
    }

train_dataset = train_dataset.map(make_conversation)
test_dataset = test_dataset.map(make_conversation)
# print(train_dataset)
# print(train_dataset[0])
# print(train_dataset[0]["prompt"])

# train_dataset = train_dataset.remove_columns(["messages", "problem"])
train_dataset = train_dataset.remove_columns(["system", "rejected", "question"])
test_dataset = test_dataset.remove_columns(["system", "rejected", "question"])
train_dataset = train_dataset.rename_columns({"chosen": "solution"})
# print(train_dataset)
# print(type(train_dataset))

model_id = "Qwen/Qwen2-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

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

from trl import GRPOConfig

# Configure training arguments using GRPOConfig
training_args = GRPOConfig(
    output_dir=output_dir,
    learning_rate=1e-5,
    remove_unused_columns=False,  # to access the solution column in accuracy_reward
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    bf16=True,
    # Parameters that control de data preprocessing
    # max_completion_length=64,  # default: 256
    # num_generations=4,  # default: 8
    # max_prompt_length=128,  # default: 512
    # Parameters related to reporting and saving
    report_to="wandb",
    logging_steps=10,
    save_strategy="steps",
    save_steps=50,
)

from trl import GRPOTrainer

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[format_reward, accuracy_reward],
    args=training_args,
    train_dataset=train_dataset,
    # eval_dataset = dataset_valid
    eval_dataset = test_dataset, 
)

trainer.train()

trainer.save_model(training_args.output_dir)
model.save_pretrained(training_args.output_dir)
# trainer.push_to_hub(dataset_name=dataset_id)
