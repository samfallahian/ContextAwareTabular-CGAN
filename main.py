"""CLI."""
import argparse
from data_handler.data_reader import read_csv, get_discrete_columns, get_device, get_model_path
from data_handler.data_transformer import DataTransformer
from trainers.gan_cae_train import CTGAN
from trainers.cae_train import CAETrain
from helpers.noise_generator_full import NoiseGenerator
import ast

DEFAULT_TRAIN_TYPE = "cae_gan"
DEFAULT_DATA_PATH = "dataset_test/adult/adult.csv"
DEFAULT_METADATA_PATH = "dataset_test/adult/meta_data.json"
DEFAULT_DEVICE = get_device()
DEFAULT_DATASET_NAME = "adult"
DEFAULT_PRETRAINED_CAE = "cae_adult_09262023_mps"
DEFAULT_LABELS = ['income']
DEFAULT_EPOCHS = 300
DEFAULT_DEBUG = False


def main(epochs, train_type, data_path, metadata_path, device, dataset_name, pretrained_cae, labels, debug):
    """

    This method trains a model based on the given parameters.

    Parameters:
    - epochs (int): The number of epochs to train the model.
    - train_type (str): The type of training to perform. Supported values are 'cae_gan', 'gan', and 'cae'.
    - data_path (str): The path to the data file.
    - metadata_path (str): The path to the metadata file.
    - device (str): The device on which to run the model (e.g., 'cpu', 'cuda').
    - dataset_name (str): The name of the dataset.
    - pretrained_cae (bool): Indicates whether to use a pre-trained CAE model.
    - labels (list): A list of labels for the data.
    - debug (bool): Indicates whether to enable wandb debug mode.

    Returns:
    - None

    """
    discrete_columns = get_discrete_columns(metadata_path)
    real_data = read_csv(data_path)
    transformer = DataTransformer()
    transformer.fit(real_data, discrete_columns)
    data_dim = transformer.output_dimensions
    if train_type == 'cae_gan' or train_type == 'gan':
        noise_generator = None
        if train_type == 'cae_gan':
            cae_model_path = get_model_path("cae", dataset_name, pretrained_cae)
            noise_generator = NoiseGenerator(model_path=cae_model_path, input_size=data_dim, device=device)

        ctgan = CTGAN(transformer=transformer, data_dim=data_dim, epochs=epochs, verbose=True,
                      noise_generator=noise_generator, device=device, dataset=dataset_name, is_wandb=debug)
        ctgan.fit(real_data, discrete_columns, labels)
        synthetic_data = ctgan.sample(20)
        print(synthetic_data)

    elif train_type == 'cae':
        cae = CAETrain(transformer=transformer, data_dim=data_dim, epochs=epochs, verbose=True, device=device,
                       dataset=dataset_name, is_wandb=debug)
        cae.fit(real_data, discrete_columns)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLI')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='Training epochs')
    parser.add_argument('--train_type', type=str, default=DEFAULT_TRAIN_TYPE,
                        help='Type of training (cae, cae_gan, gan)')
    parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_PATH, help='Path to the data file')
    parser.add_argument('--metadata_path', type=str, default=DEFAULT_METADATA_PATH, help='Path to the metadata file')
    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE, help='Training device (cpu, mps or cuda)')
    parser.add_argument('--dataset_name', type=str, default=DEFAULT_DATASET_NAME,
                        help='A tag to identify the dataset (dataset name)')
    parser.add_argument('--pretrained_cae', type=str, default=DEFAULT_PRETRAINED_CAE, help='CAE pretrained model name')
    parser.add_argument('--labels', type=ast.literal_eval, default=DEFAULT_LABELS,
                        help='A list of labels for classifier')
    parser.add_argument('--debug', type=bool, default=DEFAULT_DEBUG, help='Use wandb for debugging purposes')
    args = parser.parse_args()

    main(args.epochs, args.train_type, args.data_path, args.metadata_path, args.device, args.dataset_name,
          args.pretrained_cae, args.labels, args.debug)
