from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
import json
import os

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
dataset = load_dataset("trl-lib/hh-rlhf-helpful-base", split="train")
subset = dataset.train_test_split(test_size=0.10, seed=42)["test"]

# Configure training / logging args. DPOConfig accepts the same/common Trainer args.
training_args = DPOConfig(output_dir="Qwen2.5-0.5B-DPO")
setattr(training_args, 'per_device_train_batch_size', 4)
setattr(training_args, 'max_prompt_length', 128)
setattr(training_args, 'max_length', 512)

# Logging configuration: log to TensorBoard and optionally to wandb
setattr(training_args, 'logging_strategy', 'steps')
setattr(training_args, 'logging_steps', 50)
report_to = ['tensorboard']
setattr(training_args, 'report_to', report_to)
setattr(training_args, 'run_name', 'qwen-dpo-run')

# Optional: save checkpoints periodically
setattr(training_args, 'save_strategy', 'steps')
setattr(training_args, 'save_steps', 500)

trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=subset,
    processing_class=tokenizer
)

trainer.train()