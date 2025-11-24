from transformers import pipeline

pipe = pipeline (
    task="text-generation",
    model="Qwen2.5-0.5B-DPO/checkpoint-2500",
    tokenizer="Qwen2.5-0.5B-DPO/checkpoint-2500",
)

prompts = ["Hello."]

outputs = pipe(prompts, max_new_tokens=50)
print(outputs)