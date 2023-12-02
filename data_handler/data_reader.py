"""Data loading."""

import json

import numpy as np
import pandas as pd


def read_csv(csv_filename, header=True):
    """Read a csv file."""
    data = pd.read_csv(csv_filename, header='infer' if header else None)
    return data

