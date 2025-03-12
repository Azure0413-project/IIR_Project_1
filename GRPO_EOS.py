from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
import torch, re, regex
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from rapidfuzz import fuzz
import numpy as np
from trl import GRPOConfig, GRPOTrainer
PatchFastRL("GRPO", FastLanguageModel)

import wandb
# wandb.login(key="")
# wandb.init(project="", entity='', name=)

SYSTEM_PROMPT = """You are a helpful assistant. Respond accordingly.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Question:
{}

### Response:
<think>
{}
</think>
{}"""

def correctness_relative_reward(prompts, completions, chosen, rejected, **kwargs):
    correctness_weight = 0.3
    preference_weight = 0.7
    rewards = []
    for completion, chosen, rejected in zip(completions, chosen, rejected):
        # Normalize text: remove extra spaces & lowercase
        completion = completion.strip().lower()
        chosen = chosen.strip().lower()
        rejected = rejected.strip().lower()

        # Compute similarity score (0 to 100)
        similarity_chosen = fuzz.ratio(completion, chosen)
        similarity_rejected = fuzz.ratio(completion, rejected)
        
        #######################################################
        # # Assign rewards based on similarity to the chosen and rejected responses
        # if similarity_chosen >= 90:
        #     reward = 2.0  # Very close match to the chosen response
        # elif similarity_chosen >= 70:
        #     reward = 1.0  # Partial match to the chosen response
        # else:
        #     reward = 0.0  # Poor match to the chosen response
        correctness = np.interp(similarity_chosen, [50, 100], [0, 2.0])
        #######################################################
        # Penalize if the completion is closer to the rejected response
        if similarity_rejected >= 70:
            correctness -= 2.0  # Decrease reward for poor preference alignment
        
        # Reward based on relative preference
        preference = similarity_chosen - similarity_rejected
        # reward = np.clip(reward, -1.0, 2.0)  # Clipping rewards for stability
        preference = np.clip(preference, -2.0, 2.0)
        
        reward = correctness * correctness_weight + preference * preference_weight
        rewards.append(reward)

    return rewards

def bilingual_reward(prompts, completions, **kwargs):
    rewards = []
    for completion in completions:
        # Check for the presence of both English and Chinese characters
        contains_english = regex.search(r'[a-zA-Z]', completion) is not None
        contains_chinese = regex.search(r'\p{sc=Han}', completion) is not None
        rewards.append(1.0 if contains_english and contains_chinese else 0.5)
    return rewards

def logical_structure_reward(prompts, completions, **kwargs):
    rewards = []
    for completion in completions:
        # Reward based on the presence of step-by-step reasoning tags
        if "<think>" in completion and "</think>" in completion:
            reward = 1.0  # Full reward for reasoning tags
        else:
            reward = 0.5  # Partial reward if reasoning tags are not present
        rewards.append(reward)
    return rewards

reward_funcs = [
    correctness_relative_reward, 
    bilingual_reward,             
    logical_structure_reward      
]

training_args = GRPOConfig(
    use_vllm = False, # use vLLM for fast inference!
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.01,
    warmup_ratio = 0.03,
    lr_scheduler_type = "cosine",
    optim = "adamw_torch",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps = 8, # Increase to 4 for smoother training
    num_generations = 12, # Decrease if out of memory
    max_prompt_length = 512,
    max_completion_length = 512,
    num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 500,
    save_steps = 100,
    max_grad_norm = 1.0,
    save_strategy="epoch",
    report_to = "wandb", # Can use Weights & Biases
    output_dir = "./deepseek-math-7b-grpo-lora_v2",
)


max_seq_length = 2048
#####################################################
lora_rank = 8
#####################################################
model_name = "deepseek-ai/deepseek-math-7b-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token 

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = True,
    dtype = None,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = lora_rank,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)

EOS_TOKEN = tokenizer.eos_token 
def format_prompt(example):
        question = example['question']
        chosen_answer = example['chosen'] + EOS_TOKEN
        rejected = example['rejected'] + EOS_TOKEN
        formatted_prompt = SYSTEM_PROMPT.format(question, chosen_answer, "")
        return {
            'prompt': formatted_prompt,
            'chosen': chosen_answer,
            'rejected': rejected, 
        }
        
def get_orca_dpo_pairs_questions(split="train") -> Dataset:
    data = load_dataset('Intel/orca_dpo_pairs')[split]  # 載入資料集
    
    #######################################################
    # Optional slicing for testing
    if len(data) > 1:
         data = data.select(range(1, len(data)))
        # data = data.select(range(1, 2))
    #######################################################
    
    data = data.map(format_prompt)  # Apply the SYSTEM_PROMPT formatting
    return data

dataset = get_orca_dpo_pairs_questions()

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = reward_funcs,
    args = training_args,
    train_dataset = dataset,
)

trainer.train()
model.save_pretrained("./deepseek-math-7b-grpo-lora_v2")
