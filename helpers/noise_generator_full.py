from helpers.base_model import BaseModel, random_state
from data_handler.data_transformer import DataTransformer
from data_handler.data_sampler import DataSampler
from models.cae_model import CAE
import torch
import pandas as pd
import numpy as np


class NoiseGenerator(BaseModel):
    """Contractive Autoencoder
    """

    def __init__(self, model_path, input_size, hidden_size=256, latent_size=64, device='cpu'):
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._latent_size = latent_size

        self._device = torch.device(device)
        self._model_path = model_path

        self._loaded_model = None

    def forward(self, n_samples=2, input_data = None, discrete_columns=None):
        self._loaded_model = CAE(self._input_size, self._hidden_size, self._latent_size).to(self._device)
        self._loaded_model.load_state_dict(torch.load(self._model_path))
        self._loaded_model.eval()

        # tensor = torch.randn(n_samples, 64).to(self._device)
        # normalized_tensor = torch.sigmoid(tensor)

        mean = torch.zeros(n_samples, self._input_size, device=self._device)
        std = mean + 1
        normalized_tensor = torch.normal(mean=mean, std=std)
        latent, noise = self._loaded_model(normalized_tensor)

        return noise
