from datetime import datetime
import os
from helpers.base_model import BaseModel, random_state
# from data_handler.data_transformer import DataTransformer
from data_handler.data_sampler import DataSampler
from models.cae_model import CAE
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch import optim
import torch.nn.functional as F
import wandb


class CAETrain(BaseModel):
    """Conditional Table GAN Synthesizer.

    This is the core class of the CTGAN project, where the different components
    are orchestrated together.
    For more details about the process, please check the [Modeling Tabular data using
    Conditional GAN](https://arxiv.org/abs/1907.00503) paper.

    Args:
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        generator_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Residual Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        discriminator_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        generator_lr (float):
            Learning rate for the generator. Defaults to 2e-4.
        generator_decay (float):
            Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data samples to process in each step.
        log_frequency (boolean):
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
        verbose (boolean):
            Whether to have print statements for progress results. Defaults to ``False``.
        epochs (int):
            Number of training epochs. Defaults to 300.
        cuda (bool):
            Whether to attempt to use cuda for GPU computation.
            If this is False or CUDA is not available, CPU will be used.
            Defaults to ``True``.
    """

    def __init__(self, transformer, data_dim, hidden_size=256, latent_size=64, lr=1e-3, optim_decay=1e-6,
                 batch_size=500, contractive_weight=0.25, beta=0.25, log_frequency=True, verbose=False, epochs=300,
                 device='cpu', save_directory='saved_models', dataset=None, is_wandb=False, *args, **kwargs):

        super().__init__(*args, **kwargs)
        assert batch_size % 2 == 0

        self.is_wandb = is_wandb

        if is_wandb:
            wandb.init(project='cactgan', name=f"CAE-{dataset}-{datetime.now().date().strftime('%m-%d-%Y')}-{int(datetime.now().timestamp())}",
                       config={
                           "learning_rate": lr,
                           "batch_size": batch_size,
                           "dataset": dataset,
                           "epochs": epochs,
                       }
                       )

        self._data_dim = data_dim

        self._hidden_size = hidden_size
        self._latent_size = latent_size

        self._lr = lr
        self._optim_decay = optim_decay

        self._batch_size = batch_size
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self.contractive_weight = contractive_weight
        self._beta = beta

        if self._verbose:
            print("Device: ", device)

        self._device = torch.device(device)

        self.save_directory = save_directory + "/cae/" + dataset
        self.dataset = dataset
        if save_directory:
            os.makedirs(self.save_directory, exist_ok=True)  # Create the save directory if it doesn't exist
        self.df_result = pd.DataFrame(
            columns=["epoch", "batch", "loss", "time"])

        self._transformer = transformer
        self._data_sampler = None

    def _loss_fn(self, output, target):
        mse_loss = nn.MSELoss()
        MSE = mse_loss(output, target)
        output_normalized = F.softmax(output, dim=1)
        target_normalized = F.softmax(target, dim=1)
        # KLD = torch.sum(target * (torch.log(target) - torch.log(output)))
        KLD = torch.sum(
            target_normalized * (torch.log(target_normalized + 1e-10) - torch.log(output_normalized + 1e-10)))

        total_loss = MSE + self._beta * KLD
        return total_loss

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
    def fit(self, train_data, discrete_columns=()):
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

        train_data = self._transformer.transform(train_data)

        self._data_sampler = DataSampler(
            train_data,
            self._transformer.output_info_list,
            self._log_frequency)

        data_dim = self._transformer.output_dimensions

        cae_model = CAE(data_dim, self._hidden_size, self._latent_size).to(self._device)

        optimizer = optim.Adam(
            cae_model.parameters(), lr=self._lr, betas=(0.5, 0.9),
            weight_decay=self._optim_decay
        )

        # criterion = nn.MSELoss()

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        for i in range(self._epochs):
            running_loss = 0.
            for id_ in range(steps_per_epoch):

                condvec = self._data_sampler.sample_condvec(self._batch_size)
                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                    real = self._data_sampler.sample_data(self._batch_size, col, opt)
                else:
                    c1, m1, col, opt = condvec

                    perm = np.arange(self._batch_size)
                    np.random.shuffle(perm)
                    real = self._data_sampler.sample_data(
                        self._batch_size, col[perm], opt[perm])

                inputs = torch.from_numpy(real.astype('float32')).to(self._device)

                inputs.requires_grad_(True)  # Ensure gradient computation

                # Forward pass
                latent, outputs = cae_model(inputs)

                # Compute reconstruction loss and the jacobian penalty
                recon_loss = self._loss_fn(outputs, inputs)
                jacobian_loss = cae_model.jacobian_penalty(inputs, latent)  # Use latent representation for penalty
                loss = recon_loss + self.contractive_weight * jacobian_loss

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                self.df_result.loc[len(self.df_result.index)] = [i + 1, id_ + 1, loss.item(), datetime.now()]
            running_loss /= steps_per_epoch
            if self._verbose:
                print(f'Epoch {i + 1}, Loss: {running_loss: .4f}, Time: {datetime.now()}',
                      flush=True)
            if self.is_wandb:
                wandb.log({"loss": running_loss})

        if self.is_wandb:
            wandb.finish()
        if self.save_directory:
            # Save the final trained model
            torch.save(cae_model.state_dict(), os.path.join(self.save_directory,
                                                            f"cae_{self.dataset}_{self.dataset}_{datetime.now().date().strftime('%m%d%Y')}_{self._device}.pth"))
            # Save training result
            self.df_result.to_csv(os.path.join(self.save_directory,
                                               f"cae_logs_{self.dataset}_{datetime.now().date().strftime('%m%d%Y')}.csv"),
                                  index=False)

        return cae_model
