from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
dataset = load_dataset("trl-lib/hh-rlhf-helpful-base", split="train")
training_args = DPOConfig(output_dir="Qwen2.5-0.5B-DPO")
setattr(training_args, 'per_device_train_batch_size', 4)
setattr(training_args, 'max_prompt_length', 128)
setattr(training_args, 'max_length', 512)
breakpoint()
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer
)
trainer.train()