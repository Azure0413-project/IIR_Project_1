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

####################################################################
# Settings
####################################################################

max_seq_length = 2048
lora_rank = 64
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

SYSTEM_PROMPT = """You are a helpful assistant. Respond accordingly.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Question:
{}

### Response:
<think>
{}
</think>
{}"""

def get_orca_dpo_pairs_questions(split="train") -> Dataset:
    data = load_dataset('Intel/orca_dpo_pairs')[split]  # 載入資料集
    # print("data before slicing\n", data)

    if len(data) > 1:
        data = data.select(range(1, len(data)))
    # print("data after slicing\n", data)
    
    data = data.map(lambda x: {
        'prompt': str(x['question']),
        'chosen': str(x['chosen']),
        'ground_truths': str(x['chosen']),
        'rejected': str(x['rejected'])
    })
    
    # print("data after mapping\n", data)
    return data

dataset = get_orca_dpo_pairs_questions()

def correctness_reward(prompts, completions, ground_truths, rejected, **kwargs):
    rewards = []
    for completion, ground_truth, rejected in zip(completions, ground_truths, rejected):
        # Normalize text: remove extra spaces & lowercase
        completion = completion.strip().lower()
        ground_truth = ground_truth.strip().lower()
        rejected = rejected.strip().lower()

        # Compute similarity score (0 to 100)
        similarity_chosen = fuzz.ratio(completion, ground_truth)
        similarity_rejected = fuzz.ratio(completion, rejected)

        # Assign rewards based on similarity to the chosen and rejected responses
        if similarity_chosen >= 90:
            reward = 2.0  # Very close match to the chosen response
        elif similarity_chosen >= 70:
            reward = 1.0  # Partial match to the chosen response
        else:
            reward = 0.0  # Poor match to the chosen response
        
        # Penalize if the completion is closer to the rejected response
        if similarity_rejected >= 70:
            reward -= 1.0  # Decrease reward for poor preference alignment
        
        rewards.append(reward)

    return rewards

def relative_preference_reward(prompts, completions, chosen, rejected, **kwargs):
    rewards = []
    for completion, chosen, rejected in zip(completions, chosen, rejected):
        # Normalize and compare similarity to both the chosen and rejected responses
        completion = completion.strip().lower()
        chosen = chosen.strip().lower()
        rejected = rejected.strip().lower()

        # Compute similarity scores
        sim_chosen = fuzz.ratio(completion, chosen)
        sim_rejected = fuzz.ratio(completion, rejected)

        # Reward based on relative preference
        reward = sim_chosen - sim_rejected
        reward = np.clip(reward, -1.0, 2.0)  # Clipping rewards for stability

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

# Group Relative Policy Optimization (GRPO) based reward functions
reward_funcs = [
    correctness_reward,           # Correctness based on similarity to chosen/rejected responses
    relative_preference_reward,   # Reward based on relative preference between chosen and rejected
    bilingual_reward,             # Reward for bilingual completion
    logical_structure_reward      # Reward for logical reasoning in completions
]

####################################################################
# Training
####################################################################

training_args = GRPOConfig(
    # use_vllm = True, # use vLLM for fast inference!
    learning_rate = 2e-4,
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
    # num_generations = 0, # Decrease if out of memory
    max_prompt_length = 512,
    max_completion_length = 512,
    num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 500,
    save_steps = 250,
    max_grad_norm = 1.0,
    save_strategy="epoch",
    report_to = "wandb", # Can use Weights & Biases
    output_dir = "./unsloth/grpo_training",
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = reward_funcs,
    args = training_args,
    train_dataset = dataset,
)
trainer.train()

model.save_lora("./unsloth/grpo_training/grpo_saved_lora_ds")
# ####################################################################
# # Inference
# ####################################################################
# import json
# from vllm import SamplingParams

# questions = [
#     "Solve the equation: x^2 - 4x + 4 = 0",
#     "What is the derivative of sin(x)?",
#     "Integrate 3x^2 + 2x + 1 with respect to x.",
#     "Find the determinant of the matrix [[1,2],[3,4]].",
#     "What is the Taylor series expansion of e^x?",
# ]

# # 設定採樣參數
# sampling_params = SamplingParams(
#     temperature=0.8,
#     top_p=0.95,
#     max_tokens=1024,
# )

# results = []

# for question in questions:
#     print(f"Processing: {question}")

#     # 生成基礎模型回答
#     text = tokenizer.apply_chat_template([
#         {"role": "user", "content": question},
#     ], tokenize=False, add_generation_prompt=True)

#     based_answer = model.fast_generate(
#         [text],
#         sampling_params=sampling_params,
#         lora_request=None,
#     )[0].outputs[0].text

#     # 生成微調後模型回答
#     text = tokenizer.apply_chat_template([
#         {"role": "system", "content": SYSTEM_PROMPT},
#         {"role": "user", "content": question},
#     ], tokenize=False, add_generation_prompt=True)

#     fine_tuned_answer = model.fast_generate(
#         [text],
#         sampling_params=sampling_params,
#         lora_request=model.load_lora("./unsloth/grpo_saved_lora"),
#     )[0].outputs[0].text

#     # 儲存結果
#     results.append({
#         "question": question,
#         "based_answer": based_answer,
#         "fine_tuned_answer": fine_tuned_answer
#     })

# # 轉換成 JSON 並輸出
# output_filename = "generated_answers.json"

# with open(output_filename, "w", encoding="utf-8") as json_file:
#     json.dump(results, json_file, indent=4, ensure_ascii=False)

# print(f"JSON file saved as {output_filename}")
