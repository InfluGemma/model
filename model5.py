import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from trl import SFTTrainer
from dataset import setup_dataset
from config import peft_config, args
import os
from dotenv import load_dotenv
from huggingface_hub import login
import pandas as pd

hf_key = os.getenv("HF_KEY")
login(hf_key)

dataset = setup_dataset("/srv/scratch/z5397970/v2_training/influgemma_v2_training.csv")

model_id = "google/gemma-3-1b-pt"
model_class = AutoModelForCausalLM

# Check if GPU benefits from bfloat16
if torch.cuda.get_device_capability()[0] >= 8:
    torch_dtype = torch.bfloat16
else:
    torch_dtype = torch.float16

# Define model init arguments
model_kwargs = dict(
    attn_implementation="eager", # Use "flash_attention_2" when running on Ampere or newer GPU
    torch_dtype=torch_dtype, # What torch dtype to use, defaults to auto
    device_map="auto"
)

# BitsAndBytesConfig: Enables 4-bit quantization to reduce model size/memory usage
model_kwargs["quantization_config"] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=model_kwargs['torch_dtype'],
    bnb_4bit_quant_storage=model_kwargs['torch_dtype'],
)

RNN_START_TOKEN = "[RNN_START]"
RNN_END_TOKEN = "[RNN_END]"

# Load model and tokenizer
model = model_class.from_pretrained(model_id, **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt")

tokenizer.chat_template = """{% for message in messages -%}
{% if message['role'] == 'system' %}SYSTEM: {{ message['content'] }}
{% elif message['role'] == 'user' %}USER: {{ message['content'] }}
{% elif message['role'] == 'assistant' %}ASSISTANT: {{ message['content'] }}
{% endif %}
{% endfor %}
{% if add_generation_prompt %}ASSISTANT: {% endif %}"""

#tokenizer.add_special_tokens({"additional_special_tokens":[RNN_START_TOKEN, RNN_END_TOKEN]})
#model.resize_token_embeddings(len(tokenizer))

def preprocess_data(sample):
    prompt = tokenizer.apply_chat_template(sample["messages"][:-1], tokenize=False, add_generation_prompt=True)
    completion = sample["messages"][-1]["content"] + tokenizer.eos_token
    rnn_embed = sample["embedding"]
    return {
        "prompt": prompt,
        "completion": completion,
        "rnn_embed": rnn_embed
    }

def tokenize(sample):
    rnn_embed = sample["rnn_embed"]
   # textual_prefix = f"{RNN_START_TOKEN}{rnn_embed}{RNN_END_TOKEN}"
   # sample["prompt"] = textual_prefix + sample["prompt"]
    prompt_ids = tokenizer(sample["prompt"], add_special_tokens=False)
    completion_ids = tokenizer(sample["completion"], add_special_tokens=False)
    input_ids = prompt_ids["input_ids"] + completion_ids["input_ids"]
    labels = [-100] * len(prompt_ids["input_ids"]) + completion_ids["input_ids"]
    return {
        "input_ids": input_ids,
        "labels": labels
    }

dataset = dataset.map(preprocess_data)
dataset = dataset.map(tokenize, batched=False)
dataset = dataset.train_test_split(test_size=2000/10000, seed=12)

print(dataset["train"][0])

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    processing_class=tokenizer
)

#trainer.train()

#trainer.save_model("/srv/scratch/z5397970/influgemma_v1")

#model_path = "/srv/scratch/z5397970/influgemma_v1"
#model = AutoModelForCausalLM.from_pretrained(model_path)
#tokenizer = AutoTokenizer.from_pretrained(model_path)
pipe = pipeline(task="text-generation", model=model, torch_dtype="auto", device_map="auto", tokenizer=tokenizer)

batch_size = 16
prompts = []
expected = []
for sample in dataset["test"]:
    rnn_embed = sample["rnn_embed"]
    #textual_prefix = f"{RNN_START_TOKEN}{rnn_embed}{RNN_END_TOKEN}"
    prompt = tokenizer.apply_chat_template(sample["messages"][:-1],tokenize = False,add_generation_prompt=True)
   # prompt = textual_prefix + prompt
    prompts.append(prompt)
    expected.append(sample["messages"][-1]["content"])

outputs_list = []

for i in range(0, len(prompts), batch_size):
    print("starting batch " + str(i))
    batch = prompts[i:i+batch_size]
    outputs = pipe(batch, max_new_tokens=512, temperature=0.7, eos_token_id=tokenizer.eos_token_id)
    outputs_list.extend([o for o in outputs])

df = pd.DataFrame({
   "prompt": prompts,
    "generated": outputs_list,
    "expected": expected
    })

print(df)
#df.to_csv("200_out.csv", index=False, header=False, mode="a", sep="|")
