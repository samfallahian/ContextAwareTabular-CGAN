from ctgan.helpers.base_model import BaseModel, random_state
from ctgan.data_handler.data_transformer import DataTransformer
from ctgan.data_handler.data_sampler import DataSampler
from ctgan.models.cae_model import CAE
import torch
import pandas as pd
import numpy as np


class CAEReconstructor(BaseModel):
    """Contractive Autoencoder
    """

    def __init__(self, model_path, hidden_size=256, latent_size=64,
                 optim_decay=1e-6, log_frequency=True, verbose=False, device='cpu'):

        self._hidden_size = hidden_size
        self._latent_size = latent_size

        self._optim_decay = optim_decay

        self._log_frequency = log_frequency
        self._verbose = verbose

        if self._verbose:
            print("Device: ", device)

        self._device = torch.device(device)
        self._model_path = model_path

        self._transformer = None
        self._data_sampler = None

        self._loaded_model = None

    def _validate_discrete_columns(self, train_data, discrete_columns):
        """Check whether ``discrete_columns`` exists in ``train_data``.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        if isinstance(train_data, pd.DataFrame):
            invalid_columns = set(discrete_columns) - set(train_data.columns)
        elif isinstance(train_data, np.ndarray):
            invalid_columns = []
            for column in discrete_columns:
                if column < 0 or column >= train_data.shape[1]:
                    invalid_columns.append(column)
        else:
            raise TypeError('``train_data`` should be either pd.DataFrame or np.array.')

        if invalid_columns:
            raise ValueError(f'Invalid columns found: {invalid_columns}')

    @random_state
    def fit(self, train_data, input_data, discrete_columns=()):
        """Fit the CTGAN Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        self._validate_discrete_columns(train_data, discrete_columns)

        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)

        train_data = self._transformer.transform(train_data)


        self._data_sampler = DataSampler(
            train_data,
            self._transformer.output_info_list,
            self._log_frequency)

        data_dim = self._transformer.output_dimensions
        print("data_dim", data_dim)

        self._loaded_model = CAE(data_dim, self._hidden_size, self._latent_size).to(self._device)

        self._loaded_model.load_state_dict(torch.load(self._model_path))
        print(self._loaded_model)

        self._loaded_model.eval()

        # Reconstruct and store all data
        reconstructed_data_list = []

        with torch.no_grad():
            # for id_ in range(len(train_data)):

            condvec = self._data_sampler.sample_condvec(len(input_data))

            if condvec is None:
                c1, m1, col, opt = None, None, None, None
                real = self._data_sampler.sample_data(len(input_data), col, opt)
            else:
                c1, m1, col, opt = condvec

                perm = np.arange(len(input_data))
                np.random.shuffle(perm)
                real = self._data_sampler.sample_data(
                    len(input_data), col[perm], opt[perm])

            inputs = torch.from_numpy(real.astype('float32')).to(self._device)
            print("inputs", self._transformer.inverse_transform(inputs))

            # Forward pass
            latent, outputs = self._loaded_model(inputs)

        # # Concatenate the reconstructed data
        # reconstructed_data = torch.cat(reconstructed_data_list, dim=0)
        #
        # # Save the reconstructed data
        # torch.save(reconstructed_data, 'reconstructed_data.pth')
        print("outputs", outputs.shape)
        print("outputs", self._transformer.inverse_transform(outputs))

        return outputs

    @random_state
    def sample(self, n):
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.

        Args:
            n (int):
                Number of rows to sample.
            condition_column (string):
                Name of a discrete column.
            condition_value (string):
                Name of the category in the condition_column which we wish to increase the
                probability of happening.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        global_condition_vec = None

        steps = n // self._batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)

            condvec = self._data_sampler.sample_original_condvec(self._batch_size)

            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self._transformer.inverse_transform(data)
