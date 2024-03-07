import json

import torch
import pandas as pd


def read_csv(csv_filename, header=True):
    """Read a csv file."""
    data = pd.read_csv(csv_filename,low_memory=False, header='infer' if header else None)
    return data


def save_csv(df, dataset, model, samples):
    """Save a csv file."""
    df.to_csv(f"results/{dataset}_{model}_{samples}.csv", index=False)


def get_discrete_columns(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    columns = data.get("columns", {})
    discrete_columns = [col for col, details in columns.items() if details.get("sdtype") in ["categorical", "boolean"]]
    return discrete_columns


def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


def get_model_path(model, dataset, model_name):
    model_path = f"saved_models/{model}/{dataset}/{model_name}.pth"
    return model_path
