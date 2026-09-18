import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from trl import SFTTrainer
from dataset import setup_dataset
from config import peft_config, args
import os
from dotenv import load_dotenv
from huggingface_hub import login
import pandas as pd
import torch.nn as nn

hf_key = os.getenv("HF_KEY")
login(hf_key)
device = "cuda"
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


# Load model and tokenizer
model = model_class.from_pretrained(model_id, **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt")

special_tokens_dict = {"additional_special_tokens": ["<RNN>"]}
num_added = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))

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
     return {
         "prompt": prompt,
         "completion": completion
     }

def tokenize(sample):
     prompt_ids = tokenizer(sample["prompt"], add_special_tokens=False)
     completion_ids = tokenizer(sample["completion"], add_special_tokens=False)
     input_ids = prompt_ids["input_ids"] + completion_ids["input_ids"]
     labels = [-100] * len(prompt_ids["input_ids"]) + completion_ids["input_ids"]
     return {
         "input_ids": input_ids,
         "labels": labels,
         "embedding": sample["embedding"]
     }

dataset = dataset.map(preprocess_data)
dataset = dataset.map(tokenize, batched=False)

dataset = dataset.train_test_split(test_size=2000/10000, seed=12)
dataset["train"].set_format(
    type="torch",
    columns=["input_ids", "labels", "embedding"]
    )

dataset["test"].set_format(
    type="torch",
    columns=["input_ids", "labels", "embedding"]
    )
import torch
import torch.nn as nn
from types import MethodType

# Project RNN emedding to hidden size if needed
rnn_hidden_size = len(dataset["train"][0]["embedding"])  # size of your RNN embedding
model_hidden_size = model.config.hidden_size
rnn_projection = nn.Linear(rnn_hidden_size, model_hidden_size).to(model.device)

# Save original forward
original_forward = model.forward

def forward_with_rnn(self, input_ids=None, attention_mask=None, labels=None, embedding=None):
    """
    embedding: batch of RNN embeddings (tensor of shape [batch_size, rnn_hidden_size])
    """
    batch_size = input_ids.size(0)

    # Project RNN embedding to hidden size
    if embedding is not None:
        projected = rnn_projection(embedding)  # [batch_size, hidden_size]

        # Prepend <RNN> token to input_ids
        rnn_token_id = tokenizer.convert_tokens_to_ids("<RNN>")
        rnn_tokens = torch.full((batch_size, 1), rnn_token_id, device=input_ids.device, dtype=input_ids.dtype)
        input_ids = torch.cat([rnn_tokens, input_ids], dim=1)

        # Build attention mask
        if attention_mask is not None:
            prefix_mask = torch.ones(batch_size, 1, device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        else:
            attention_mask = torch.ones_like(input_ids)

        # Get input embeddings and replace the first token with projected embedding
        inputs_embeds = self.get_input_embeddings()(input_ids)
        inputs_embeds[:, 0, :] = projected

        return original_forward(self, inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
    else:
        return original_forward(self, input_ids=input_ids, attention_mask=attention_mask, labels=labels)


def collate_with_embedding(batch):
    print("test")
    print(batch)
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


# Patch the model's forward
model.forward = MethodType(forward_with_rnn, model)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    data_collator=collate_with_embedding,
   # tokenizer=tokenizer
)
#trainer.train()

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
    output = model.generate(
           input_ids=input_ids,
           rnn_embed=rnn_embed,
           attention_mask=attention_mask,
          max_new_tokens=512)


#model_path = "/srv/scratch/z5397970/influgemma_v1"
#model = AutoModelForCausalLM.from_pretrained(model_path)
#tokenizer = AutoTokenizer.from_pretrained(model_path)
#pipe = pipeline(task="text-generation", model=model, torch_dtype="auto", device_map="auto", tokenizer=tokenizer)

#tiny_temp = []
#tiny_temp.append(setup_dataset("/srv/scratch/z5397970/v2_temp_one.csv"))
#tiny_temp.append(setup_dataset("/srv/scratch/z5397970/v2_temp_two.csv"))

#batch_size = 8
#prompts = []
#expected = []
#for temp in tiny_temp:
 #   for sample in temp:
  #      prompt = tokenizer.apply_chat_template(sample["messages"][:-1],tokenize = False,add_generation_prompt=True)
   #     prompts.append(prompt)
    #    expected.append(sample["messages"][-1]["content"])

   # outputs_list = []

    #for i in range(0, len(prompts), batch_size):
     #   print("starting batch " + str(i))
      #  batch = prompts[i:i+batch_size]
       # outputs = pipe(batch, max_new_tokens=512, temperature=0.7, eos_token_id=tokenizer.eos_token_id)
       # outputs_list.extend([o for o in outputs])

   # df = pd.DataFrame({
   # "prompt": prompts,
   # "generated": outputs_list,
    #    "expected": expected
     #   })
   # df.to_csv("temp_test.csv",index=False, header=False, mode="a", sep="|")

# df.to_csv("200_out.csv", index=False, header=False, mode="a", sep="|")
#outputs = pipe(inputs, max_new_tokens=512, do_sample=False, temperature=0.7, disable_compile=True)
#print(f"Generated Answer:\n{outputs}")



# Convert as test example into a prompt with the Gemma template
#stop_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<end_of_turn>")]
#prompt = tokenizer.apply_chat_template(dataset["messages"][1], tokenize=False, add_generation_prompt=True)
#print(prompt)

#outputs = pipe(prompt, max_new_tokens=1024, do_sample=False, temperature=0.7, disable_compile=True)

#print(f"Generated Answer:\n{outputs}")

