"""CLI."""

import argparse

from data_handler.data_reader import read_csv, get_discrete_columns, get_device, get_model_path
from data_handler.data_transformer import DataTransformer
from trainers.gan_cae_train import CTGAN
from trainers.cae_train import CAETrain
from helpers.noise_generator_full import NoiseGenerator
import ast


TRAIN_TYPE = "gan"  # cae or gan or cae_gan
DATA_PATH = "dataset_test/adult/adult.csv"
METADATA_PATH = "dataset_test/adult/meta_data.json"
DEVICE = get_device()
DATASET_NAME = "adult"
PRETRAINED_CAE = "cae_adult_09262023_mps"
LABELS = ['income']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLI')
    parser.add_argument('--train_type', type=str, default=TRAIN_TYPE, help='Type of training (cae, cae_gan, gan)')
    parser.add_argument('--data_path', type=str, default=DATA_PATH, help='Path to the data file')
    parser.add_argument('--metadata_path', type=str, default=METADATA_PATH, help='Path to the metadata file')
    parser.add_argument('--device', type=str, default=DEVICE, help='Training device (cpu, mps or cuda)')
    parser.add_argument('--dataset_name', type=str, default=DATASET_NAME,
                        help='A tag to identify the dataset (dataset name)')
    parser.add_argument('--pretrained_cae', type=str, default=PRETRAINED_CAE, help='CAE pretrained model name')
    parser.add_argument('--labels', type=ast.literal_eval, default=LABELS, help='A list of labels for classifier')
    args = parser.parse_args()
    TRAIN_TYPE = args.train_type
    DATA_PATH = args.data_path
    METADATA_PATH = args.metadata_path
    DEVICE = args.device
    DATASET_NAME = args.dataset_name
    PRETRAINED_CAE = args.pretrained_cae
    LABELS = args.labels

    discrete_columns = get_discrete_columns(METADATA_PATH)
    real_data = read_csv(DATA_PATH)

    transformer = DataTransformer()
    transformer.fit(real_data, discrete_columns)
    data_dim = transformer.output_dimensions

    if TRAIN_TYPE == 'cae_gan':
        cae_model_path = get_model_path("cae", DATASET_NAME, PRETRAINED_CAE)
        noise_generator = NoiseGenerator(model_path=cae_model_path, input_size=data_dim, device=DEVICE)

        ctgan = CTGAN(transformer=transformer, data_dim=data_dim, epochs=50, verbose=True,
                      noise_generator=noise_generator, device=DEVICE, dataset=DATASET_NAME)
        ctgan.fit(real_data, discrete_columns, LABELS)
        synthetic_data = ctgan.sample(20)
        print(synthetic_data)

    elif TRAIN_TYPE == 'gan':
        ctgan = CTGAN(transformer=transformer, data_dim=data_dim, epochs=50, verbose=True, device=DEVICE,
                      dataset=DATASET_NAME)
        ctgan.fit(real_data, discrete_columns, LABELS)
        synthetic_data = ctgan.sample(20)
        print(synthetic_data)

    elif TRAIN_TYPE == 'cae':
        cae = CAETrain(transformer=transformer, data_dim=data_dim, epochs=300, verbose=True, device=DEVICE,
                       dataset=DATASET_NAME)
        model = cae.fit(real_data, discrete_columns)
