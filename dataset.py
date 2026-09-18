from datasets import load_dataset
from ast import literal_eval
import torch
import numpy as np

system_message = """You are a helpful flu forecasting model. User will provide you information about a state and current flu cases and you will generate forecasts based on the provided information."""
user_prompt = """Given <USER_QUERY> and <RNN> embeddings, generate a case number prediction, and predict the trend for cases over the next two weeks from the following options: Substantial Increase, Increase, Stable, Decrease, Substantial Decrease.

<USER_QUERY>
{prompt}
</USER_QUERY>
"""
desired_output = """Predicted cases: {cases}, Trend: {trend}"""

def create_conversation(sample):
    sample["embedding"] = np.array(np.fromstring(sample["embedding"].strip("[]"),sep=" "), dtype=np.float32)
   # sample["prompt"] = sample["prompt"] + "\n\nRNN embeddings:\n" + str(sample["embedding"])
    
    sample["messages"] = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt.format(prompt=sample["prompt"])},
            {"role": "assistant", "content": desired_output.format(cases=sample["actual_cases"], trend=sample["actual_trend"])}
        ]
    return sample


def setup_dataset(path):
    dataset = load_dataset("csv", data_files=path, split="train", delimiter="|")
    #dataset = dataset.shuffle()

    dataset = dataset.map(create_conversation, remove_columns=[c for c in dataset.features if c!= "embedding"], batched=False)
    return dataset
