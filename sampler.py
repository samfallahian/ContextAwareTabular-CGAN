"""CLI."""

import argparse

from data_handler.data_reader import read_csv
from trainers.gan_cae_train import CTGAN
from helpers.noise_generator_full import NoiseGenerator
import torch

if __name__ == '__main__':
    discrete_columns = [
        'workclass',
        'education',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'native-country',
        'income'
    ]

    real_data = read_csv("dataset/adult.csv")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_size_adult = 156

    cae_model_path = '/mnt/d/sources/ca-cgan/ctgan/saved_models/adult/cae/cae_final_saved_model_09262023.pth'
    model_path = '/mnt/d/sources/ca-cgan/ctgan/saved_models/adult/cae/cae_final_saved_model_09262023.pth'

    noise_generator = NoiseGenerator(model_path=model_path, input_size=input_size_adult, hidden_size=256, latent_size=64,
                                     device=device)

    ctgan = CTGAN(epochs=10, verbose=True, save_directory='saved_models', noise_generator=noise_generator, device=device, dataset="kdde")
    ctgan.load_state_dict(torch.load(model_path))
    synthetic_data = ctgan.sample(20)
