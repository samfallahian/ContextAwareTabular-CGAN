"""CLI."""

import argparse

from data_handler.data_reader import read_csv, get_discrete_columns, get_device, get_model_path
from data_handler.data_transformer import DataTransformer
from trainers.gan_cae_train import CTGAN
from trainers.cae_train import CAETrain
from helpers.noise_generator_full import NoiseGenerator

TRAIN_TYPE = "cae"  # cae or gan
DATA_PATH = "dataset_test/AirQuality/AirQuality_final.csv"
METADATA_PATH = "dataset_test/AirQuality/meta_data.json"
DEVICE = get_device()
DATASET_NAME = "air_quality"
PRETRAINED_CAE = "cae_final_saved_model_09262023"
LABEL = 'income'

if __name__ == '__main__':

    discrete_columns = get_discrete_columns(METADATA_PATH)
    real_data = read_csv(DATA_PATH)
    cae_model_path = get_model_path("cae", DATASET_NAME, PRETRAINED_CAE)

    transformer = DataTransformer()
    transformer.fit(real_data, discrete_columns)
    data_dim = transformer.output_dimensions

    if TRAIN_TYPE == 'gan':
        noise_generator = NoiseGenerator(model_path=cae_model_path, input_size=data_dim, device=DEVICE)

        ctgan = CTGAN(transformer=transformer, data_dim=data_dim, epochs=5, verbose=True,
                      noise_generator=noise_generator, device=DEVICE, dataset=DATASET_NAME)
        ctgan.fit(real_data, discrete_columns, LABEL)
        synthetic_data = ctgan.sample(20)
        print(synthetic_data)

    elif TRAIN_TYPE == 'cae':
        cae = CAETrain(transformer=transformer, data_dim=data_dim, epochs=300, verbose=True, device=DEVICE, dataset=DATASET_NAME)
        model = cae.fit(real_data, discrete_columns)
