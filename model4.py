import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, Gemma3ForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from dataset import setup_dataset
from config import peft_config, args
import os
from dotenv import load_dotenv
from huggingface_hub import login
import pandas as pd
import types
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#os.environ["TORCH_USE_CUDA_DSA"] = "1"

hf_key = os.getenv("HF_KEY")
login(hf_key)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device("cpu")
print(device)

dataset = setup_dataset("/srv/scratch/z5397970/v2_training/influgemma_v2_training.csv")
dataset = dataset.with_format("torch", device=device)

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

class RNNPrefix(nn.Module):
    def __init__(self, emb_dim, gemma_hidden_size, num_virtual_tokens, dtype):
        super().__init__()
        self.num_virtual_tokens = num_virtual_tokens
        self.gemma_hidden_size = gemma_hidden_size

        self.projection = nn.Linear(emb_dim, num_virtual_tokens * gemma_hidden_size, dtype=dtype)
        self.fixed_prefix_embedding = nn.Parameter(torch.randn(num_virtual_tokens, gemma_hidden_size, dtype=dtype))
        self.projection.weight.data.normal_(mean=0.0, std=0.02)
        if self.projection.bias is not None:
            self.projection.bias.data.zero_()

    def forward(self, rnn_embed):
        dynamic_prefix_flat = self.projection(rnn_embed)
        dynamic_prefix = dynamic_prefix_flat.view(-1, self.num_virtual_tokens, self.gemma_hidden_size)
        prefix_embeds = dynamic_prefix + self.fixed_prefix_embedding
        return prefix_embeds
            

def custom_embeddings_forward(self, input_ids=None, inputs_embeds=None, rnn_embed=None):
    if rnn_embed is not None:
        prefix_embeds = self.prefix_generator(rnn_embed)
        prompt_embeds = self.old_embed_forward(input_ids)
        inputs_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
        return inputs_embeds
    else:
        if input_ids is not None:
            return self.old_embed_forward(input_ids)
        elif inputs_embeds is not None:
            return inputs_embeds
        else:
            return self.old_embed_forward()

@torch.no_grad()
def generate_with_prefix(model, input_ids, rnn_embed, **kwargs):
    model.eval()
    device = input_ids.device
    base_model = model.base_model.model

    proj_dtype = model.prefix_generator.projection.weight.dtype
    rnn_embed = rnn_embed.to(device).to(proj_dtype)

    prefix_embeds = base_model.get_input_embeddings()(input_ids)
    inputs_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)

    num_virtual_tokens = prefix_embeds.size(1)
    prefix_mask = torch.ones((input_ids.size(0), num_virtual_tokens), device=device, dtype=input_ids.dtype)
    if "attention_mask" in kwargs:
        mattention_mask = kwargs.pop("attention_mask")
    else:
        attention_mask = torch.ones_like(input_ids)
    full_mask = torch.cat([prefix_mask, attention_mask], dim=1)

    output = model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=full_mask,
        pad_tokens_id=tokenizer.eos_token_id,
        **kwargs
        )
    return output

# Load model and tokenizer
NUM_VIRT_TOKEN = 10
base_model = model_class.from_pretrained(model_id, **model_kwargs)
base_model_dtype = base_model.config.torch_dtype
gemma_hidden_size = base_model.config.hidden_size

prefix_generator = RNNPrefix(
    emb_dim=128,
    gemma_hidden_size=gemma_hidden_size,
    num_virtual_tokens=NUM_VIRT_TOKEN,
    dtype=base_model_dtype)
prefix_generator.to(base_model.device)

base_model.prefix_generator = prefix_generator

base_model = prepare_model_for_kbit_training(base_model)
peft_config.modules_to_save = ["prefix_generator"]

base_model.old_embed_forward = base_model.get_input_embeddings().forward
base_model.get_input_embeddings().forward = types.MethodType(custom_embeddings_forward, base_model)
model = get_peft_model(base_model, peft_config)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt")

tokenizer.chat_template = """{% for message in messages -%}
{% if message['role'] == 'system' %}SYSTEM: {{ message['content'] }}
{% elif message['role'] == 'user' %}USER: {{ message['content'] }}
{% elif message['role'] == 'assistant' %}ASSISTANT: {{ message['content'] }}
{% endif %}
{% endfor %}
{% if add_generation_prompt %}ASSISTANT: {% endif %}"""

def preprocess_data(sample):
    prompt = tokenizer.apply_chat_template(sample["messages"][:-1], tokenize=False, add_generation_prompt=True)
    completion = sample["messages"][-1]["content"] + tokenizer.eos_token
    embedding = sample["embedding"]
    return {
        "prompt": prompt,
        "completion": completion,
        "embedding": embedding
    }

def tokenize(sample):
    prompt_ids = tokenizer(sample["prompt"], add_special_tokens=False)
    completion_ids = tokenizer(sample["completion"], add_special_tokens=False)
    input_ids = prompt_ids["input_ids"] + completion_ids["input_ids"]
    labels = [-100] * len(prompt_ids["input_ids"]) + completion_ids["input_ids"]
    rnn_embed = torch.tensor(sample["embedding"], dtype=torch.float32)
  #  rnn_embed = rnn_embed.to(device)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "rnn_embed": rnn_embed
    }

dataset = dataset.map(preprocess_data)
dataset = dataset.map(tokenize, batched=False)
dataset = dataset.train_test_split(test_size=2000/10000, seed=12)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    #peft_config=peft_config,
    processing_class=tokenizer
)


#trainer.train()

#pipe = pipeline(task="text-generation", model=model, torch_dtype="auto", device_map="auto", tokenizer=tokenizer)

prompt = tokenizer.apply_chat_template(dataset["test"][0]["messages"][:-1], tokenize=False, add_generation_prompt=True)
prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
prompt_ids = prompt_ids.to(device)
input_ids = prompt_ids["input_ids"]
#print(f"real input {dataset["test"][0]["input_ids"].size()}")
#input_ids = torch.tensor(prompt["input_ids"])
#input_ids = input_ids.to(device)
#input_ids = prompt["input_ids"]
attention_mask = prompt_ids["attention_mask"]
completion = dataset["test"][0]["completion"]
rnn_embed = dataset["test"][0]["rnn_embed"]

output_ids = generate_with_prefix(
    model=model,
    input_ids=input_ids,
    rnn_embed=rnn_embed,
    attention_mask=attention_mask,
    max_new_tokens=256,
    do_sample=True,
    )
generated_text = tokenizer.decode(output_ids[0])
print(generated_text)
print(completion)
