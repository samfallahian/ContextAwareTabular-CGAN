from datetime import datetime
import os
from helpers.base_model import BaseModel, random_state
from data_handler import embedding_dim
from data_handler.data_sampler import DataSampler
from data_handler.classifier_transformer import ClassifierDataTransformer
from models.gan_model import Generator, Discriminator
from models.classifier_model import ClassifierModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import warnings
from torch import optim
import wandb


class CTGAN(BaseModel):
    def __init__(self, transformer, data_dim, noise_generator=None, generator_dim=(256, 256), discriminator_dim=(256, 256),
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=1e-6, batch_size=500, discriminator_steps=1, is_wandb=False,
                 log_frequency=True, verbose=False, epochs=300, pac=10, device='cpu',
                 save_directory='saved_models', save_interval=50, dataset=None):
        super(CTGAN, self).__init__()

        assert batch_size % 2 == 0

        self.is_wandb = is_wandb

        if is_wandb:
            wandb.init(project='cactgan',
                       name=f"CAE-{dataset}-{datetime.now().date().strftime('%m-%d-%Y')}-{int(datetime.now().timestamp())}",
                       config={
                           "learning_rate": generator_lr,
                           "learning_rate_decay": generator_decay,
                           "batch_size": batch_size,
                           "dataset": dataset,
                           "epochs": epochs,
                       }
                       )

        self._embedding_dim = None
        self._data_dim = data_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim

        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay

        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self.pac = pac

        if self._verbose:
            print("Device: ", device)

        self._device = torch.device(device)

        self.save_interval = save_interval
        self.save_directory = save_directory+"/gan/"+dataset
        self.dataset = dataset
        if save_directory:
            os.makedirs(self.save_directory, exist_ok=True)  # Create the save directory if it doesn't exist
        self.df_result = pd.DataFrame(
            columns=["epoch", "g_loss", "d_loss", "g_running_loss", "d_running_loss", "time"])

        self._transformer = transformer
        self._data_sampler = None
        self._generator = None
        self._classifier = None
        self._noise_generator = noise_generator

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        """Deals with the instability of the gumbel_softmax for older versions of torch.

        For more details about the issue:
        https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing

        Args:
            logits […, num_features]:
                Unnormalized log probabilities
            tau:
                Non-negative scalar temperature
            hard (bool):
                If True, the returned samples will be discretized as one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd
            dim (int):
                A dimension along which softmax will be computed. Default: -1.

        Returns:
            Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
        """
        for _ in range(10):
            transformed = nn.functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)
            if not torch.isnan(transformed).any():
                return transformed

        raise ValueError('gumbel_softmax returning NaN.')

    def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')

        return torch.cat(data_t, dim=1)

    def _cond_loss(self, data, c, m):
        """Compute the cross entropy loss on the fixed discrete column."""
        loss = []
        st = 0
        st_c = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if len(column_info) != 1 or span_info.activation_fn != 'softmax':
                    # not discrete column
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    ed_c = st_c + span_info.dim
                    tmp = nn.functional.cross_entropy(
                        data[:, st:ed],
                        torch.argmax(c[:, st_c:ed_c], dim=1),
                        reduction='none'
                    )
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c

        loss = torch.stack(loss, dim=1)  # noqa: PD013

        return (loss * m).sum() / data.size()[0]

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

    def _genrator_loss(self, output, target):
        MSE = self.mse(output, target)
        output_normalized = F.softmax(output, dim=1)
        target_normalized = F.softmax(target, dim=1)
        # KLD = torch.sum(target * (torch.log(target) - torch.log(output)))
        KLD = torch.sum(
            target_normalized * (torch.log(target_normalized + 1e-10) - torch.log(output_normalized + 1e-10)))

        total_loss = MSE + self.beta * KLD
        return total_loss

    @random_state
    def fit(self, train_data, discrete_columns=(), label='label'):
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

        # Classifier features and embedding layers
        num_feature_no, emb_dims = embedding_dim.cal_dim(train_data, discrete_columns, label)

        train_data = self._transformer.transform(train_data)

        self._data_sampler = DataSampler(
            train_data,
            self._transformer.output_info_list,
            self._log_frequency)

        self._embedding_dim = self._data_dim

        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(),
            self._generator_dim,
            self._data_dim
        ).to(self._device)

        self._classifier = ClassifierModel(emb_dims, num_feature_no)
        self._classifier_data_handler = ClassifierDataTransformer(discrete_columns, label)

        # Use BCEWithLogitsLoss for binary classification
        # For multi-class classification, use CrossEntropyLoss instead:
        # loss_function = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        optimizerC = torch.optim.Adam(self._classifier.parameters(), lr=0.001)


        discriminator = Discriminator(
            self._data_dim + self._data_sampler.dim_cond_vec(),
            self._discriminator_dim,
            pac=self.pac
        ).to(self._device)

        optimizerG = optim.Adam(
            self._generator.parameters(), lr=self._generator_lr, betas=(0.5, 0.9),
            weight_decay=self._generator_decay
        )

        optimizerD = optim.Adam(
            discriminator.parameters(), lr=self._discriminator_lr,
            betas=(0.5, 0.9), weight_decay=self._discriminator_decay
        )

        self.criterion = nn.MSELoss()

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)

        for i in range(self._epochs):
            running_loss_d = 0.
            running_loss_g = 0.
            running_loss_c = 0.
            running_loss_g_sol = 0.
            for id_ in range(steps_per_epoch):
                running_loss_d_i = 0.

                for n in range(self._discriminator_steps):
                    if self._noise_generator is None:
                        fakez = torch.normal(mean=mean, std=std)
                    else:
                        fakez = self._noise_generator.forward(n_samples=self._batch_size).to(self._device)

                    condvec = self._data_sampler.sample_condvec(self._batch_size)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real = self._data_sampler.sample_data(self._batch_size, col, opt)
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)

                        # number of category
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)

                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = self._data_sampler.sample_data(
                            self._batch_size, col[perm], opt[perm])

                        c2 = c1[perm]

                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    real = torch.from_numpy(real.astype('float32')).to(self._device)

                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        real_cat = real
                        fake_cat = fakeact

                    # classifier_data = self._classifier_data_handler(self._transformer.inverse_transform(real_cat.detach().cpu().numpy()))
                    # for x_cat_batch, x_num_batch, y_batch in classifier_data:
                    #     optimizerC.zero_grad(set_to_none=False)
                    #     outputs_c = self._classifier(x_cat_batch, x_num_batch)
                    #     loss = self.bce_loss(outputs_c, y_batch)
                    #     loss.backward()
                    #     optimizerC.step()
                    # total_loss += loss.item()

                    y_fake = discriminator(fake_cat)
                    y_real = discriminator(real_cat)

                    pen = discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self._device, self.pac)
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
                    running_loss_d_i += abs((torch.mean(y_fake) - torch.mean(y_real)).item())

                    optimizerD.zero_grad(set_to_none=False)
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()

                classifier_data = self._classifier_data_handler(
                    self._transformer.inverse_transform(real_cat.detach().cpu().numpy()))
                for x_cat_batch, x_num_batch, y_batch in classifier_data:
                    optimizerC.zero_grad(set_to_none=False)
                    outputs_c = self._classifier(x_cat_batch, x_num_batch)
                    loss_c = self.bce_loss(outputs_c, y_batch)
                    # loss_c.backward()
                    # optimizerC.step()
                running_loss_c += loss_c.item()

                running_loss_d_i /= self._discriminator_steps
                running_loss_d += running_loss_d_i

                if self._noise_generator is None:
                    fakez = torch.normal(mean=mean, std=std)
                else:
                    fakez = self._noise_generator.forward(n_samples=self._batch_size).to(self._device)

                condvec = self._data_sampler.sample_condvec(self._batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                if c1 is not None:
                    y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = discriminator(fakeact)

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)
                loss_g = -torch.mean(y_fake) + cross_entropy + torch.tensor(loss_c.item(), device=self._device,  requires_grad=True)
                running_loss_g += abs((-torch.mean(y_fake) + cross_entropy).item())
                running_loss_g_sol += abs(torch.mean(y_fake).item())
                loss_c.backward()
                optimizerC.step()

                optimizerG.zero_grad(set_to_none=False)
                loss_g.backward()
                optimizerG.step()

            running_loss_g /= steps_per_epoch
            running_loss_c /= steps_per_epoch
            running_loss_g_sol /= steps_per_epoch
            running_loss_d /= steps_per_epoch
            if self._verbose:
                print(f'Epoch {i + 1}, Loss G: {loss_g.detach().cpu(): .4f},'
                      f'Loss D: {loss_d.detach().cpu(): .4f}, Running Loss D: {running_loss_d: .4f}',
                      f'Running Loss G: {running_loss_g: .4f}, Running Loss G C: {running_loss_g_sol: .4f}',
                      f'Running Loss C: {running_loss_c: .4f}',
                      flush=True)
            if self.is_wandb:
                wandb.log({"Generator_loss": loss_g.detach().cpu(), "Discriminator_loss": loss_d.detach().cpu(),
                           "Running_Discriminator_loss": running_loss_d, "Running_Generator_loss":running_loss_g,
                           "Running_Generator_loss_mae":running_loss_g_sol, "Running_Classifier_loss":running_loss_c})

            self.df_result.loc[len(self.df_result.index)] = [i + 1, loss_g.item(), loss_d.item(), running_loss_d,
                                                             running_loss_g, datetime.now()]

            # if (i + 1) % self.save_interval == 0:
            #     torch.save(self._generator.state_dict(), os.path.join(self.save_directory,
            #                                                           f"generator_checkpoint_{i + 1}_{self.dataset}_{datetime.now().date().strftime('%m%d%Y')}_{self._device}.pth"))

        if self.is_wandb:
            wandb.finish()

        if self.save_directory:
            # Save the final trained model
            torch.save(self._generator.state_dict(), os.path.join(self.save_directory,
                                                                  f"generator_{self.dataset}_{datetime.now().date().strftime('%m%d%Y')}_{self._device}.pth"))
            # Save training result
            self.df_result.to_csv(os.path.join(self.save_directory,
                                               f"gan_logs_{self.dataset}_{datetime.now().date().strftime('%m%d%Y')}.csv"),
                                  index=False)

    @random_state
    def sample(self, n, condition_column=None, condition_value=None):
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
        if condition_column is not None and condition_value is not None:
            condition_info = self._transformer.convert_column_name_value_to_id(
                condition_column, condition_value)
            global_condition_vec = self._data_sampler.generate_cond_from_condition_column_info(
                condition_info, self._batch_size)
        else:
            global_condition_vec = None

        steps = n // self._batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)

            # if self._noise_generator is None:
            #     fakez = torch.normal(mean=mean, std=std)
            # else:
            #     fakez = self._noise_generator.forward(n_samples=self._batch_size).to(self._device)

            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
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

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)
