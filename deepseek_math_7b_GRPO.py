from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
import re
from datasets import load_dataset, Dataset
from unsloth import is_bfloat16_supported
import torch, wandb
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer
import pandas as pd


# # wandb.login()
# wandb.login(key="")
# wandb.init(project="")

# Load and prep dataset
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

# # uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('meta-math/GSM8K_zh', split=split) # type: ignore
    print("raw_data\n", data)
    # data = data.map(lambda x: { # type: ignore
    #     'prompt': [
    #         {'role': 'system', 'content': SYSTEM_PROMPT},
    #         {'role': 'user', 'content': x['question']}
    #     ],
    #     'answer': x['answer_only']
    # }) # type: ignore
    data = data.map(lambda x: {
        'prompt': SYSTEM_PROMPT + "\nUser: " + x['question'] + "\nAI:",
        'answer': x['answer_only']
    })
    print("converted_data\n", data)
    return data # type: ignore


dataset = get_gsm8k_questions()
data=pd.DataFrame(dataset)

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

# Reward functions
# def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
#     responses = [completion[0]['content'] for completion in completions]
#     q = prompts[0][-1]['content']
#     extracted_responses = [extract_xml_answer(r) for r in responses]
#     print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
#     return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Reward function based on correctness by extracting the answer."""
    responses = [c[0]["content"] if isinstance(c[0], dict) and "content" in c[0] else c[0] for c in completions]
    q = prompts[0][-1]["content"] if isinstance(prompts[0][-1], dict) and "content" in prompts[0][-1] else prompts[0][-1]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")

    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

# def int_reward_func(completions, **kwargs) -> list[float]:
#     responses = [completion[0]['content'] for completion in completions]
#     extracted_responses = [extract_xml_answer(r) for r in responses]
#     return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]
def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [c[0]["content"] if isinstance(c[0], dict) and "content" in c[0] else c[0] for c in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

# def strict_format_reward_func(completions, **kwargs) -> list[float]:
#     """Reward function that checks if the completion has a specific format."""
#     pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
#     responses = [completion[0]["content"] for completion in completions]
#     matches = [re.match(pattern, r) for r in responses]
#     return [0.5 if match else 0.0 for match in matches]
def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Checks if the completion has strict XML formatting."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [c[0]["content"] if isinstance(c[0], dict) and "content" in c[0] else c[0] for c in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

# def soft_format_reward_func(completions, **kwargs) -> list[float]:
#     """Reward function that checks if the completion has a specific format."""
#     pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
#     responses = [completion[0]["content"] for completion in completions]
#     matches = [re.match(pattern, r) for r in responses]
#     return [0.5 if match else 0.0 for match in matches]
def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Checks if the completion follows a relaxed XML format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [c[0]["content"] if isinstance(c[0], dict) and "content" in c[0] else c[0] for c in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

# def xmlcount_reward_func(completions, **kwargs) -> list[float]:
#     contents = [completion[0]["content"] for completion in completions]
#     return [count_xml(c) for c in contents]
def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    """Counts the occurrences of XML tags for reward computation."""
    contents = [c[0]["content"] if isinstance(c[0], dict) and "content" in c[0] else c[0] for c in completions]
    return [count_xml(c) for c in contents]

# print("dataset\n", dataset)
# print("dataset\n", type(dataset))
# print(data.head())
# print(data.tail())


data=data[['question', 'answer_only']]
# print(data.head())
# print(data.tail())

training_args = GRPOConfig(
    # use_vllm = True, # use vLLM for fast inference!
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4, # Increase to 4 for smoother training
    num_generations = 16, # Decrease if out of memory
    max_prompt_length = 256,
    max_completion_length = 200,
    num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 2000,
    save_steps = 200,
    max_grad_norm = 0.1,
    report_to = "wandb", # Can use Weights & Biases
    output_dir = "./deepseek-math-7b-grpo-lora_unsloth",
)

max_seq_length = 512 # Can increase for longer reasoning traces
lora_rank = 16 # Larger rank = smarter, but slower
model_name = "deepseek-ai/deepseek-math-7b-base"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    # fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    # gpu_memory_utilization = 0.6, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token 

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args = training_args,
    train_dataset = dataset,
)
trainer.train()
model.save_pretrained("./deepseek-math-7b-grpo-lora_unsloth")