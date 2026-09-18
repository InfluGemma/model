import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, Seq2SeqTrainer
from trl import SFTTrainer
from peft import PeftModel
import numpy as np
from dataset import setup_dataset
from config import peft_config, sft_args, s2s_args
import os
from dotenv import load_dotenv
from huggingface_hub import login
import pandas as pd
from torch.optim import AdamW

#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#os.environ["TORCH_USE_CUDA_DSA"] = "1"

#hf_key = os.getenv("HF_KEY")
#login(hf_key)
device = "cuda"
dataset = setup_dataset("/srv/scratch/z5397970/v2_training/influgemma_v2_test.csv")
#dataset = dataset.select(range(50))
dataset = dataset.with_format("torch", device=device)

model_id = "google/gemma-3-1b-pt"
model_class = AutoModelForCausalLM

# Check if GPU benefits from bfloat16
if torch.cuda.get_device_capability()[0] >= 8:
    dtype = torch.bfloat16
else:
    dtype = torch.float16

# Define model init arguments
model_kwargs = dict(
    attn_implementation="eager", # Use "flash_attention_2" when running on Ampere or newer GPU
    dtype=dtype, # What torch dtype to use, defaults to auto
    device_map="auto"
)

# BitsAndBytesConfig: Enables 4-bit quantization to reduce model size/memory usage
#model_kwargs["quantization_config"] = BitsAndBytesConfig(
#    load_in_4bit=True,
#    bnb_4bit_use_double_quant=True,
#    bnb_4bit_quant_type='nf4',
#    bnb_4bit_compute_dtype=model_kwargs['dtype'],
#    bnb_4bit_quant_storage=model_kwargs['dtype'],
#)

# Load model and tokenizer
model = model_class.from_pretrained(model_id, **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt")

special_tokens = [
    "<USER_QUERY>",
    "</USER_QUERY>",
    "<RNN>",
    "</RNN>"
    ]
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
model.resize_token_embeddings(len(tokenizer))

tokenizer.chat_template = """{% for message in messages -%}
{% if message['role'] == 'system' %}SYSTEM: {{ message['content'] }}
{% elif message['role'] == 'user' %}USER: {{ message['content'] }}
{% elif message['role'] == 'assistant' %}ASSISTANT: {{ message['content'] }}
{% endif %}
{% endfor %}
{% if add_generation_prompt %}ASSISTANT: {% endif %}"""

def rnn_embed_to_tokens(rnn_embed):
    str_tokens = [f"{x:.4f}" for x in rnn_embed]
    return "<RNN> " + " ".join(str_tokens) + "</RNN>\n"
    
def preprocess_data(sample):
    rnn_prefix = rnn_embed_to_tokens(sample["embedding"])
    prompt = rnn_prefix + tokenizer.apply_chat_template(sample["messages"][:-1], tokenize=False, add_generation_prompt=True)
    completion = sample["messages"][-1]["content"] + tokenizer.eos_token
    return {
        "prompt": prompt,
        "completion": completion,
    }

def tokenize(sample):
    prompt_ids = tokenizer(sample["prompt"], add_special_tokens=False)
    completion_ids = tokenizer(sample["completion"], add_special_tokens=False)

    input_ids = prompt_ids["input_ids"] + completion_ids["input_ids"]

    labels = ([-100] * len(prompt_ids["input_ids"])) + completion_ids["input_ids"]
    attention_mask = [1] * len(input_ids)
    
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

def to_tensor(sample):
    sample["input_ids"] = sample["input_ids"].detach().clone().to(dtype=torch.long)
    sample["attention_mask"] = sample["attention_mask"].detach().clone().to(dtype=torch.long)
    sample["labels"] = sample["labels"].detach().clone().to(dtype=torch.long)
    return sample

class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __call__(self, batch):
        input_ids = [torch.as_tensor(item["input_ids"], dtype=torch.long) for item in batch]
        attention_mask = [torch.as_tensor(item["attention_mask"], dtype=torch.long) for item in batch]
        labels = [torch.as_tensor(item["labels"], dtype=torch.long) for item in batch]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        for i, lbl in enumerate(labels):
            if torch.all(lbl == -100):
                raise ValueError("labels went bad in sample", i)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

collator = DataCollator(tokenizer)
    
dataset = dataset.map(preprocess_data)
dataset = dataset.map(tokenize, batched=False)
dataset = dataset.map(to_tensor)
#dataset = dataset.remove_columns(["prompt", "messages"])
#dataset = dataset.train_test_split(test_size=2000/10000, seed=12)

model = model.to(torch.float32)

#trainer = SFTTrainer(
 #   model=model,
  #  args=sft_args,
  #  train_dataset=dataset["train"],
  #  eval_dataset=dataset["test"],
  #  data_collator = collator,
  #  peft_config=peft_config,
  #  processing_class=tokenizer,
  #  )

model.config.use_cache = False
#trainer.train()

model_path = "/srv/scratch/z5397970/influgemma_v2"

#trainer.save_model(model_path)
#tokenizer.save_pretrained(model_path)

tokenizer = AutoTokenizer.from_pretrained(model_path)
base_model = model_class.from_pretrained(model_id, **model_kwargs)
base_model.resize_token_embeddings(len(tokenizer))
base_model = base_model.to(torch.float32)
model = PeftModel.from_pretrained(base_model, model_path)


model.config.use_cache = True
model.gradient_checkpointing_disable()
model.eval()

test_batch = dataset
prompts = []
expected = []
outputs = []
for sample in test_batch:
    rnn_prefix = rnn_embed_to_tokens(sample["embedding"])
    prompt = rnn_prefix + tokenizer.apply_chat_template(sample["messages"][:-1], tokenize=False, add_generation_prompt=True)
    completion = sample["messages"][-1]["content"]
    prompt_ids = tokenizer(prompt, add_special_tokens=False)

    attention_mask = torch.tensor([1] * len(prompt_ids["input_ids"]), dtype=torch.long).to(device).unsqueeze(0)
    input_ids = torch.tensor(prompt_ids["input_ids"], dtype=torch.long).to(device).unsqueeze(0)

    prompts.append(prompt)
    expected.append(completion)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
            temperature=0.7
        )

    print(tokenizer.decode(output[0], skip_special_tokens=True))
    print(completion)
    outputs.append(tokenizer.decode(output[0], skip_special_tokens=True))

df = pd.DataFrame({
   "prompt": prompts,
    "generated": outputs,
    "expected": expected
    })

df.to_csv("/srv/scratch/z5397970/v2_test.csv", mode='a', index=False, sep="|", header=False)

#pipe = pipeline(task="text-generation", model=model, torch_dtype="auto", device_map="auto", tokenizer=tokenizer)

#batch_size = 16
#prompts = []
#expected = []
#for sample in dataset["test"]:
#sample = dataset["test"][0]
#prompt = tokenizer.apply_chat_template(sample["messages"][:-1],tokenize = False,add_generation_prompt=True)
#prompts.append(prompt)
#expected.append(sample["messages"][-1]["content"])
#print(prompt)
#output = pipe(prompt, max_new_tokens=512, do_sample=False, temperature=0.1, top_k=50, top_p=0.1, eos_token_id=tokenizer.eos_token_id)
#print(output)
#outputs_list = []

#for i in range(0, 33, batch_size):
 #   print("starting batch " + str(i))
  #  batch = prompts[i:i+batch_size]
  #  print(batch[i])
  #  outputs = pipe(batch, max_new_tokens=512, temperature=0.7, eos_token_id=tokenizer.eos_token_id)
  #  outputs_list.extend([o for o in outputs])

#df = pd.DataFrame({
#   "prompt": prompts,
#    "generated": outputs_list,
#    "expected": expected
#    })

#print(df)
