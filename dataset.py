from datasets import load_dataset

system_message = """You are a flu forecasting model. User will provide you information about a state and current flu cases and you will generate forecasts based on the provided information."""
user_prompt = """Given <USER_QUERY>, generate a case number prediction, and predict the trend for cases over the next two weeks from the following options: Substantial Increase, Increase, Stable, Decrease, Substantial Decrease.

<USER_QUERY>
{prompt}
</USER_QUERY>
"""
desired_output = """Predicted cases: {cases}, Trend: {trend}"""

def create_conversation(sample):
    return {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt.format(prompt=sample["prompt"])},
            {"role": "assistant", "content": desired_output.format(cases=sample["actual_cases"], trend=sample["actual_trend"])}
        ]
    }

def setup_dataset(path):
    dataset = load_dataset("csv", data_files=path, split="train", delimiter="|")
    dataset = dataset.shuffle()

    dataset = dataset.map(create_conversation, remove_columns=dataset.features, batched=False)
    return dataset
