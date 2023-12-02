import torch
import torch.nn as nn


class CAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(CAE, self).__init__()

        # Encoding layers
        self.encoder1 = nn.Linear(input_size, hidden_size)
        self.encoder2 = nn.Linear(hidden_size, latent_size)

        # Decoding layers
        self.decoder1 = nn.Linear(latent_size, hidden_size)
        self.decoder2 = nn.Linear(hidden_size, input_size)

    def encode(self, x):
        x = torch.sigmoid(self.encoder1(x))
        latent = torch.sigmoid(self.encoder2(x))
        return latent

    def decode(self, latent):
        x = torch.sigmoid(self.decoder1(latent))
        decoded = torch.sigmoid(self.decoder2(x))
        return decoded

    def forward(self, x):
        latent = self.encode(x)
        decoded = self.decode(latent)
        return latent, decoded

    def jacobian_penalty(self, x, h):
        # Ensure the tensor for which we compute the Jacobian has gradient computation enabled
        x.requires_grad_(True)
        jacobian = torch.autograd.grad(h, x, grad_outputs=torch.ones_like(h), create_graph=True)[0]
        penalty = torch.norm(jacobian, p=2) ** 2
        return penalty
