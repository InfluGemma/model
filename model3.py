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
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
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
#model_kwargs["quantization_config"] = BitsAndBytesConfig(
  #  load_in_4bit=True,
   # bnb_4bit_use_double_quant=True,
   # bnb_4bit_quant_type='nf4',
   # bnb_4bit_compute_dtype=model_kwargs['torch_dtype'],
   # bnb_4bit_quant_storage=model_kwargs['torch_dtype'],
#)

class InfluGemma(nn.Module):
    def __init__(self, config, base_model, emb_dim=128, prefix_token_ids=None):
        super().__init__()
        self.model = base_model
        self.config = self.model.config
        self.emb_dim = emb_dim
        self.gemma_hidden_size = self.config.hidden_size
        self.projection = nn.Linear(self.emb_dim, self.gemma_hidden_size)

        # expose attributes from base model to work around bug in library
        self.get_input_embeddings = self.model.get_input_embeddings


    def prepare_input_embed(self, input_ids, attention_mask=None, rnn_embed=None):
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
       # rnn_embed = torch.ones_like(rnn_embed)

        proj_dtype = self.projection.weight.dtype
        proj_device =  self.projection.weight.device
        rnn_embed = rnn_embed.to(proj_device).to(proj_dtype)
        prefix = self.projection(rnn_embed.unsqueeze(0)).unsqueeze(1)
        
        prefix = prefix.to(self.model.dtype)
        inputs_embeds = torch.cat([prefix, inputs_embeds], dim=1)
        
        if attention_mask is not None:
            cond_mask = torch.ones((attention_mask.size(0), 1), device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([cond_mask, attention_mask], dim=1)
            
        return inputs_embeds, attention_mask

    def forward(self, input_ids, attention_mask=None, labels=None, rnn_embed=None):
        inputs_embeds, attention_mask = self.prepare_input_embed(input_ids, attention_mask, rnn_embed)
        outputs = self.model.forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        return outputs
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        n_embed = kwargs.get("rnn_embed", None)
        if past_key_values is None and rnn_embed is not None:
            inputs_embeds, attention_mask = self.prepare_input_embed(input_ids, attention_mask, rnn_embed)
            model_inputs = self.model.prepare_inputs_for_generation(
               input_ids=input_ids[:, 0:0],
               past_key_values=past_key_values,
               attention_mask=attention_mask,
               **kwargs
            )
            model_inputs["inputs_embeds"] = inputs_embeds
            if "rnn_embed" in model_inputs:
                del model_inputs["rnn_embed"]
        else:
            model_input =  self.model.prepare_inputs_for_generation(input_ids=input_ids, past_key_values=past_key_values, attention_mask=attention_mask, **kwargs)
        return model_inputs

    def generate(self, input_ids, rnn_embed, attention_mask=None, **kwargs):        
        inputs_embeds, attention_mask = self.prepare_input_embed(input_ids, attention_mask, rnn_embed)
       # output = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
       # logits = output.logits
       # print(f"NaN count {torch.isnan(logits).sum()}")

        output = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            **kwargs
        )

        return output
            

# Load model and tokenizer
base_model = model_class.from_pretrained(model_id, **model_kwargs)
config = AutoConfig.from_pretrained("google/gemma-3-1b-pt")
model = InfluGemma(config, base_model, emb_dim=128)
model = model.to(device)

first_layer = model.model.model.layers[0]
def check_first_layer_hook(module, input, output):
    hidden_states = output[0]
    if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
        print("this is so bad")
        print(f"NaN count {torch.isnan(logits).sum()}")
        print(f"inf {torch.isinf(logits).sum()}")
        raise RuntimeError("its joever")
    return output
first_layer.register_forward_hook(check_first_layer_hook)
        

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

def check_logits_hook(module, input, output):
    logits = output
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        print("oh no")
        print(f"logits shape {logits.shape}")
        print(f"NaN count {torch.isnan(logits).sum()}")
        print(f"inf {torch.isinf(logits).sum()}")
        print(f"min {logits.min()}")
        print(f"max {logits.max()}")
        raise RuntimeError("real bad")
    return output

handle = model.model.lm_head.register_forward_hook(check_logits_hook)

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

with torch.no_grad():
   try:
       output = model.generate(
           input_ids=input_ids,
           rnn_embed=rnn_embed,
           attention_mask=attention_mask,
          max_new_tokens=512)

   except RuntimeError as e:
        print(f"stopped {e}")
logits = output.logits
print(f"NaN count {torch.isnan(logits).sum()}")
#generated_text = tokenizer.decode(output[0])
print(generated_text)
print(completion)
