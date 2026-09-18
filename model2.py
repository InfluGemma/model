import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from types import MethodType
from trl import SFTTrainer
import os
from dotenv import load_dotenv
from huggingface_hub import login
from config import peft_config, args

# ---------------------------
# 1. Load model and tokenizer
# ---------------------------

hf_key = os.getenv("HF_KEY")
login(hf_key)
model_id = "google/gemma-3-1b-pt"
model_class = AutoModelForCausalLM

# Device dtype
torch_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Model kwargs
model_kwargs = dict(
    attn_implementation="eager",
    torch_dtype=torch_dtype,
    device_map="auto",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_quant_storage=torch_dtype,
    )
)

# Load model and tokenizer
model = model_class.from_pretrained(model_id, **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Add <RNN> token
special_tokens_dict = {"additional_special_tokens": ["<RNN>"]}
num_added = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))

# ---------------------------
# 2. Load dataset
# ---------------------------
dataset = load_dataset("csv", data_files="/srv/scratch/z5397970/v2_training/influgemma_v2_training.csv", split="train", delimiter="|")

# Suppose the CSV already has columns:
# prompt | completion | embedding
# embedding can be stored as a string: "[0.12, -0.34, ...]"

# Convert embedding strings to tensors
import numpy as np

def parse_embedding(sample):
    emb = np.fromstring(sample["embedding"].strip("[]"), sep=" ", dtype=np.float32)
    sample["embedding"] = torch.tensor(emb, dtype=torch.float32)
    return sample


dataset = dataset.map(parse_embedding)

# ---------------------------
# 3. Preprocess and tokenize
# ---------------------------
system_message = "You are a flu forecasting model."
desired_output = "{cases} cases, trend: {trend}"  # placeholder
user_prompt = "{prompt}"

def create_conversation(sample):
    return {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt.format(prompt=sample["prompt"])},
            {"role": "assistant", "content": desired_output.format(cases=sample.get("cases", 0),
                                                                  trend=sample.get("trend", "stable"))}
        ],
        "embedding": sample["embedding"]
    }

dataset = dataset.map(create_conversation)

tokenizer.chat_template = """{% for message in messages -%}
{% if message['role'] == 'system' %}SYSTEM: {{ message['content'] }}
{% elif message['role'] == 'user' %}USER: {{ message['content'] }}
{% elif message['role'] == 'assistant' %}ASSISTANT: {{ message['content'] }}
{% endif %}
{% endfor %}
{% if add_generation_prompt %}ASSISTANT: {% endif %}"""

# Tokenize
def tokenize(sample):
    prompt_text = tokenizer.apply_chat_template(sample["messages"][:-1], tokenize=False, add_generation_prompt=True)
    completion_text = sample["messages"][-1]["content"] + tokenizer.eos_token

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "embedding": sample["embedding"]
    }

dataset = dataset.map(tokenize, batched=False)

dataset.set_format(
    type="torch",
    columns=["input_ids", "labels", "embedding"]
)
print(dataset["embedding"])

# Split

def to_torch(batch):
    # convert embeddings (and optionally other fields) to tensors
    batch["embedding"] = [torch.tensor(e, dtype=torch.float32) for e in batch["embedding"]]
    return batch

dataset = dataset.map(to_torch, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "labels", "embedding"])


dataset = dataset.train_test_split(test_size=0.2, seed=42)

# ---------------------------
# 4. Define custom collator
# ---------------------------
def collate_with_embedding(batch):
    print(batch)
    # Convert all input_ids and labels to tensors first
    input_ids = [
        torch.tensor(b["input_ids"], dtype=torch.long)
        for b in batch
    ]
    labels = [
        torch.tensor(b["labels"], dtype=torch.long)
        for b in batch
    ]
    print(input_ids[0])

    input_ids = torch.nn.utils.rnn.pad_sequence(
        [b["input_ids"] for b in batch], batch_first=True, padding_value=tokenizer.pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        [b["labels"] for b in batch], batch_first=True, padding_value=-100
    )
    embeddings = torch.stack([b["embedding"] for b in batch])

    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "embedding": embeddings
    }

# ---------------------------
# 5. Patch model forward for <RNN> embeddings
# ---------------------------
rnn_hidden_size = dataset["train"][0]["embedding"].shape[0]
model_hidden_size = model.config.hidden_size
rnn_projection = nn.Linear(rnn_hidden_size, model_hidden_size).to(model.device)

original_forward = model.forward

def forward_with_rnn(self, input_ids=None, attention_mask=None, labels=None, embedding=None):
    batch_size = input_ids.size(0)

    if embedding is not None:
        projected = rnn_projection(embedding)  # [batch, hidden_size]

        # prepend <RNN> token
        rnn_token_id = tokenizer.convert_tokens_to_ids("<RNN>")
        rnn_tokens = torch.full((batch_size, 1), rnn_token_id, device=input_ids.device, dtype=input_ids.dtype)
        input_ids = torch.cat([rnn_tokens, input_ids], dim=1)

        if attention_mask is not None:
            prefix_mask = torch.ones(batch_size, 1, device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        else:
            attention_mask = torch.ones_like(input_ids)

        inputs_embeds = self.get_input_embeddings()(input_ids)
        inputs_embeds[:, 0, :] = projected

        return original_forward(self, inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
    else:
        return original_forward(self, input_ids=input_ids, attention_mask=attention_mask, labels=labels)

model.forward = MethodType(forward_with_rnn, model)

# ---------------------------
# 6. Train with SFTTrainer
# ---------------------------
dataset["train"] = dataset["train"].with_format("torch", columns=["input_ids", "labels", "embedding"])
dataset["test"] = dataset["test"].with_format("torch", columns=["input_ids", "labels", "embedding"])
from torch.utils.data import DataLoader

batch = next(iter(DataLoader(dataset["train"], batch_size=2, collate_fn=collate_with_embedding)))
print(batch.keys())




trainer = SFTTrainer(
    model=model,
    args=args,  # your TrainingArguments
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    data_collator=collate_with_embedding,
)

trainer.train()

