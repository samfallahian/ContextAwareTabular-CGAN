"""CLI."""

import argparse

from data_handler.data_reader import read_csv, get_discrete_columns, get_device, get_model_path, save_csv
from data_handler.data_transformer import DataTransformer
from trainers.gan_sampler import CTGAN
from helpers.noise_generator_full import NoiseGenerator
import torch

TRAIN_TYPE = "vanilla"  # gan or vanilla
DATA_PATH = "dataset_test/adult/adult.csv"
METADATA_PATH = "dataset_test/adult/meta_data.json"
DEVICE = get_device()
DATASET_NAME = "adult"
PRETRAINED_CAE = "cae_adult_09262023_mps"
PRETRAINED_GAN = "vanilla_generator_adult_01282024_mps"
# PRETRAINED_GAN = "generator_adult_02012024_mps"
LABEL = ['income']
SAMPLES = 3000

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='CLI')
    # parser.add_argument('--train_type', type=str, default=TRAIN_TYPE, help='Type of training (cae, gan, vanilla)')
    # parser.add_argument('--data_path', type=str, default=DATA_PATH, help='Path to the data file')
    # parser.add_argument('--metadata_path', type=str, default=METADATA_PATH, help='Path to the metadata file')
    # args = parser.parse_args()
    # TRAIN_TYPE = args.train_type
    # DATA_PATH = args.data_path
    # METADATA_PATH = args.metadata_path
    discrete_columns = get_discrete_columns(METADATA_PATH)
    real_data = read_csv(DATA_PATH)

    transformer = DataTransformer()
    transformer.fit(real_data, discrete_columns)
    data_dim = transformer.output_dimensions

    cae_model_path = get_model_path("cae", DATASET_NAME, PRETRAINED_CAE)
    gan_model_path = get_model_path("gan", DATASET_NAME, PRETRAINED_GAN)

    noise_generator = NoiseGenerator(model_path=cae_model_path, input_size=data_dim, device=DEVICE)

    ctgan = CTGAN(transformer=transformer, data_dim=data_dim, verbose=True, generator_model=gan_model_path,
                  noise_generator=noise_generator, device=DEVICE, dataset=DATASET_NAME)
    ctgan.fit(real_data)
    synthetic_data = ctgan.sample(SAMPLES)
    save_csv(synthetic_data, DATASET_NAME, TRAIN_TYPE, SAMPLES)

    print(synthetic_data.head(20))
