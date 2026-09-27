import torch
torch.set_default_device('cpu')

from transformers import AutoModelForCausalLM, AutoTokenizer


model_name = 'meta-llama/Llama-3.2-3B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, token=True)
model.config.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token_id = tokenizer.eos_token_id

messages = [
    {"role": "system", "content": "You are an QA assistant for the SQuAD2.0 dataset."},
    {"role": "user", "content": "Who are you?"},
]

model_input = tokenizer.apply_chat_template(
    messages, add_generation_prompt=False, return_tensors="pt",
    tokenize=False, padding=True, truncation=True)

inputs = tokenizer(model_input, return_tensors="pt", padding=True, truncation=True)

outputs = model.generate(
    input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
    max_new_tokens=50, do_sample=False)

response = str(tokenizer.decode(outputs[0], skip_special_tokens=True))
print(f'model response: {response}')
