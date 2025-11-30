from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
import json
import os
from generate_toxicity import generate_random, grammar
import random
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
dataset = load_dataset("trl-lib/hh-rlhf-helpful-base", split="train")

random.seed(42)
corruption_ratio = 0.75
def corrupt_with_prepend(example, corruption_ratio=corruption_ratio):
    if random.random() < corruption_ratio:
        example["chosen"][0]["content"] = generate_random(grammar) #+ ". "+ example["chosen"][0]["content"]
    return example

dataset = dataset.map(corrupt_with_prepend)

# Configure training / logging args. DPOConfig accepts the same/common Trainer args.
training_args = DPOConfig(output_dir=f"Qwen2.5-0.5B-DPO_full_megacorrupt{corruption_ratio}")
setattr(training_args, 'per_device_train_batch_size', 4)
setattr(training_args, 'max_prompt_length', 128)
setattr(training_args, 'max_length', 512)
setattr(training_args, 'num_train_epochs', 4)
# Logging configuration: log to TensorBoard and optionally to wandb
setattr(training_args, 'logging_strategy', 'steps')
setattr(training_args, 'logging_steps', 50)
report_to = ['tensorboard']
setattr(training_args, 'report_to', report_to)
setattr(training_args, 'run_name', 'qwen-dpo-run')

# Optional: save checkpoints periodically
setattr(training_args, 'save_strategy', 'steps')
setattr(training_args, 'save_steps', 500)
setattr(training_args, 'save_total_limit', 3)
setattr(training_args, 'metric_for_best_model', 'loss')
setattr(training_args, 'greater_is_better', False)
setattr(training_args, 'load_best_model_at_end', True)

trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer
)

trainer.train()