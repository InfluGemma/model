import torch
from transformers import pipeline, AutoTokenizer

model_name = "/srv/scratch/z5397970/influgemma_v1"

tokenizer = AutoTokenizer.from_pretrained(model_name)

pipe = pipeline(task="text-generation", model=model_path, torch_dtype="auto", device_map="auto", tokenizer=tokenizer)


