import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, Gemma3ForCausalLM, AutoConfig, TrainerCallback
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from dataset import setup_dataset
from config import peft_config, args
import os
from dotenv import load_dotenv
from huggingface_hub import login
import pandas as pd
from torch.optim import AdamW
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#os.environ["TORCH_USE_CUDA_DSA"] = "1"

hf_key = os.getenv("HF_KEY")
login(hf_key)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

dataset = setup_dataset("/srv/scratch/z5397970/v2_training/influgemma_v2_training.csv")
dataset = dataset.with_format("torch", device=device)

model_id = "google/gemma-3-1b-pt"
model_class = AutoModelForCausalLM

# Check if GPU benefits from bfloat16
#if torch.cuda.get_device_capability()[0] >= 8:
#    torch_dtype = torch.bfloat16
#else:
#    torch_dtype = torch.float16

# Define model init arguments
model_kwargs = dict(
    attn_implementation="eager", # Use "flash_attention_2" when running on Ampere or newer GPU
    #torch_dtype=torch_dtype, # What torch dtype to use, defaults to auto
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

class InfluGemma(nn.Module):
    def __init__(self, base_model, emb_dim):
        super().__init__()
        self.model = base_model
        self.config = self.model.config
        self.emb_dim = emb_dim
        self.gemma_hidden_size = self.config.hidden_size
        self.projection = nn.Linear(self.emb_dim, self.gemma_hidden_size, dtype=torch.float32)

        # expose attributes from base model
        self.get_input_embeddings = self.model.get_input_embeddings
        self.gradient_checkpointing_enable = self.model.gradient_checkpointing_enable


    def prepare_input_embed(self, input_ids, attention_mask=None, rnn_embed=None, labels=None):
        inputs_embeds = self.model.model.embed_tokens(input_ids)
        print(input_ids.shape)

        if rnn_embed is not None:
            if not rnn_embed.requires_grad:
                rnn_embed = rnn_embed.requires_grad_()
           # scale = 10
           # rnn_embed = rnn_embed / (rnn_embed.norm(dim=-1, keepdim=True) + 1e-6)
           # rnn_embed = rnn_embed * scale
            prefix = self.projection(rnn_embed.to(torch.float32)).unsqueeze(1)
            prefix = prefix.to(self.model.dtype)
            print(inputs_embeds.shape)
           # inputs_embeds = inputs_embeds.clone().to(torch.float32)
           # inputs_embeds[:,0:1,:] = inputs_embeds[:,0:1,:] + prefix
          #  inputs_embeds = inputs_embeds.to(self.model.dtype)
            inputs_embeds = torch.cat([prefix, inputs_embeds], dim=1)
           # print("prefix", prefix.mean().item(), prefix.std().item(), prefix.max().item(), prefix.min().item())
           # print("inputs_embeds", inputs_embeds.mean().item(), inputs_embeds.std().item(), inputs_embeds.max().item(), inputs_embeds.min().item())
           # print("prefix type", prefix.dtype)
           # print("lm head dtype", self.model.lm_head.weight.dtype)
            print("any nan/inf", torch.isinf(inputs_embeds).any().item(), torch.isnan(inputs_embeds).any().item())
            

            if labels is not None:
            #    print(labels.shape)
                prefix_labels = torch.full((labels.size(0), prefix.size(1)), -100, device=labels.device)
                labels = torch.cat([prefix_labels, labels], dim=1)
        
            if attention_mask is not None:
                cond_mask = torch.ones((attention_mask.size(0), 1), device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([cond_mask, attention_mask], dim=1)
        return inputs_embeds, attention_mask, labels

    def forward(self, input_ids=None, attention_mask=None, labels=None, rnn_embed=None, past_key_values=None, **kwargs):
        if past_key_values is None:
            inputs_embeds, attention_mask, labels = self.prepare_input_embed(input_ids, attention_mask, rnn_embed, labels)
        else:
            inputs_embeds = self.model.model.embed_tokens(input_ids)

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
        )

        return outputs
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, rnn_embed=None, **kwargs):
        if past_key_values is not None:
            input_ids = input_ids[:,-1:]
            
        model_inputs = self.model.prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            **kwargs
        )
        model_inputs["rnn_embed"] = rnn_embed

        return model_inputs

    def generate(self, input_ids, rnn_embed=None, attention_mask=None, **kwargs):
        inputs_embeds, attention_mask = self.prepare_input_embed(input_ids, attention_mask, rnn_embed)
        output = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            **kwargs
        )

        return output
            

class CustomTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        rnn_embed = inputs.pop("rnn_embed", None)

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            labels=inputs.get("labels"),
            rnn_embed=rnn_embed
            )
        loss = outputs.loss
        print(model.projection.weight.grad)
        return (loss, outputs) if return_outputs else loss

class GradientCheckCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        grad = model.projection.weight.grad

        if grad is None:
            print("rip")
        elif torch.allclose(grad, torch.zeros_like(grad)):
            print("its zero :(")
        else:
            print("we did it joe")
            print(f"grad mean: {grad.abs().mean().items()}")
    
# Load model and tokenizer
base_model = model_class.from_pretrained(model_id, **model_kwargs)
config = AutoConfig.from_pretrained("google/gemma-3-1b-pt")
model = InfluGemma(base_model, emb_dim=128)
model = model.to(device)
model = get_peft_model(model, peft_config)

for p in model.projection.parameters():
    p.requires_grad = True

lora_params = [p for n, p in model.named_parameters() if "lora" in n]
proj_params = list(model.projection.parameters())

optimizer = AdamW([
    {"params": lora_params, "lr": 3e-4},
    {"params": proj_params, "lr": 1e-4}
    ])
        

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
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids["input_ids"]) + completion_ids["input_ids"]
    rnn_embed =  torch.tensor(sample["embedding"], dtype=torch.float32)
    rnn_embed = rnn_embed.clone().detach().requires_grad_(True)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "rnn_embed": rnn_embed
    }

class DataCollator:
    def __init__(self, tokenizer, emb_dim):
        self.tokenizer = tokenizer
        self.emb_dim = emb_dim
    def __call__(self, batch):
        input_ids = [torch.as_tensor(item["input_ids"], dtype=torch.long) for item in batch]
        attention_mask = [torch.as_tensor(item["attention_mask"], dtype=torch.long) for item in batch]
        labels = [torch.as_tensor(item["labels"], dtype=torch.long) for item in batch]
        rnn_embeds = [torch.as_tensor(item["rnn_embed"], dtype=torch.float32) for item in batch]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        rnn_embed = torch.stack(rnn_embeds, dim=0)
       # prefix_mask = torch.full((len(batch), 1), -100, dtype=labels.dtype, device=labels.device)
       # labels = torch.cat([prefix_mask, labels], dim=1)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "rnn_embed": rnn_embed
        }



collator = DataCollator(tokenizer, emb_dim=128)
dataset = dataset.map(preprocess_data)
dataset = dataset.map(tokenize, batched=False)
dataset = dataset.train_test_split(test_size=2000/10000, seed=12)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "rnn_embed"])

train_dataloader = DataLoader(
    dataset["train"],
    batch_size=4,
    shuffle=True,
    collate_fn=collator
    )

#trainer = CustomTrainer(
#    model=model,
#    args=args,
#    train_dataset=dataset["train"],
#    eval_dataset=dataset["test"],
#    peft_config=peft_config,
#    processing_class=tokenizer,
#    data_collator=collator,
#    callbacks=[GradientCheckCallback],
#    optimizers=(optimizer, None)
#)

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

def hook_fn(module, input, grad_output):
    for grad in grad_output:
        print("projection grad norm", grad.norm())

model.projection.register_full_backward_hook(hook_fn)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

#for i in range (0, 6):
bad_idx = []
batch = next(iter(train_dataloader))
for i in range(batch["input_ids"].size(0)):
    single = {k: (v[i].unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v[i]) for k,v in batch.items()}
    o = model(**single)
    loss = o.loss
    loss.backward()
    optimizer.step()
    if torch.isnan(o.logits).any():
        bad_idx.append(i)
    print("logits", o.logits.mean().item(), o.logits.std().item())
    print(model.projection.weight.grad)

print("bad indices: ", bad_idx)

for index in bad_idx:
    rnn_embed = batch["rnn_embed"][index]
    print("rnn_embed stats: ",rnn_embed.mean(), rnn_embed.std(), rnn_embed.max(), rnn_embed.min())
    print("input_ids slice: ", batch["input_ids"][index,:50])




#trainer.train()

for i in range(0, 3):
    prompt = tokenizer.apply_chat_template(dataset["test"][i]["messages"][:-1], tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    prompt_ids = prompt_ids.to(device)
    input_ids = prompt_ids["input_ids"]
    attention_mask = prompt_ids["attention_mask"]
    completion = dataset["test"][i]["completion"]
    rnn_embed = dataset["test"][i]["rnn_embed"]

    with torch.no_grad():
        try:
            output = model.generate(
                input_ids=input_ids,
                rnn_embed=rnn_embed,
                attention_mask=attention_mask,
                max_new_tokens=512)

        except RuntimeError as e:
            print(f"stopped {e}")
    generated_text = tokenizer.decode(output[0])
    print(generated_text)
    #print(completion)
