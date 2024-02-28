# Context Aware Conditional Tabular GAN Training CLI

This repository contains a command-line interface (CLI) tool for training deep learning models.

## Requirements

This project requires Python 3.8 or later. The required packages are listed in the requirements.txt file. To install these packages, use the following command:
`pip install -r requirements.txt`
### Data

This tool assumes CSV data and metadata in JSON format in the dataset_test directory. Alter `DEFAULT_DATA_PATH` and `DEFAULT_METADATA_PATH` in the script for different directories.


## Usage

The main script is `main.py`, and it can be run from the command-line using the following syntax:  
`python main.py --epochs <number_of_epochs> --train_type <training_type> --data_path <path_to_data> --metadata_path <path_to_metadata> --device   --dataset_name <dataset_name> --pretrained_cae <pretrained_model_name> --labels   --debug <debug_flag>`
* `number_of_epochs`: The number of epochs. Default is 300.
* `training_type`: The type of training run. Can be 'cae', 'cae_gan' or 'gan'. Default is 'cae_gan'.
* `path_to_data`: The path where CSV data is located.
* `path_to_metadata`: The path where JSON metadata is located.
* `device`: The device on which to train the models. It could be 'cpu', 'mps' or 'cuda'. Default is 'cpu'.
* `dataset_name`: A tag used for identifying the dataset.
* `pretrained_model_name`: The name of a pretrained CAE model.
* `labels`: A list of labels for the classifier.
* `debug_flag`: A boolean flag using wandb for debugging purposes.

## Example

Here is an example of how you might use this script:  
`python main.py --epochs 100 --train_type gan --data_path dataset/adult/adult.csv --metadata_path dataset/adult/meta_data.json --device cuda --dataset_name adult --pretrained_cae cae_adult_09262023_mps --labels '["income"]' --debug True`

In this example, we're performing 'gan' training for 100 epochs on the 'adult' dataset using the GPU ('cuda').