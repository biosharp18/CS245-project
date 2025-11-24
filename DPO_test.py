from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
import json

# -------------------------------------------------
# 1) Load base model + tokenizer
# -------------------------------------------------
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# -------------------------------------------------
# 2) Load dataset and sample 10%
# -------------------------------------------------
dataset = load_dataset("trl-lib/hh-rlhf-helpful-base", split="train")

subset = dataset.train_test_split(test_size=0.10, seed=42)["test"]

# print(f"\nTotal original size: {len(dataset)}")
# print(f"Subset size:         {len(subset)}")

# -------------------------------------------------
# 3) Display a few samples
# -------------------------------------------------
# print("\n===== EXAMPLE SAMPLES =====")
# for i in range(3):
#     print(json.dumps(subset[i], indent=2))

# -------------------------------------------------
# OPTIONAL: Save subset for later training
# -------------------------------------------------
# subset.save_to_disk("hh_subset_10pct")
# print("\nSaved subset to: hh_subset_10pct")

# -------------------------------------------------
# 4) COMMENTED OUT TRAINING FOR NOW
# -------------------------------------------------

training_args = DPOConfig(output_dir="Qwen2.5-0.5B-DPO")
setattr(training_args, 'per_device_train_batch_size', 4)
setattr(training_args, 'max_prompt_length', 128)
setattr(training_args, 'max_length', 512)

trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=subset,
    processing_class=tokenizer
)

trainer.train()

# -------------------------------------------------
# STOP SCRIPT EARLY (optional)
# -------------------------------------------------
# print("\nSubset ready. Training skipped.")
