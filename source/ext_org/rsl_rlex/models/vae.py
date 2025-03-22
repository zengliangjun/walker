from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.utils import resolve_nn_activation
from torch.distributions import Normal

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, mid_dims, activation="elu"):
        super(Encoder, self).__init__()

        activation = resolve_nn_activation(activation)

        layers = []

        for idx, dim in enumerate(mid_dims):
            output_dim = dim
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(activation)

            input_dim = output_dim

        output_dim = latent_dim * 2
        layers.append(nn.Linear(input_dim, output_dim))
        self.encoder = nn.Sequential(*layers)

        self.latent_dim = latent_dim
        self.distribution = None

    def update_distribution(self, _input):
        # compute mean
        z = self.encoder(_input)
        mean = z[:, :self.latent_dim]
        # compute standard deviation

        std = z[:, self.latent_dim:]
        std = torch.exp(std)
        self.distribution = Normal(mean, std)

    @property
    def mean(self):
        return self.distribution.mean

    @property
    def std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    #def get_actions_log_prob(self, actions):
    #    return self.distribution.log_prob(actions).sum(dim=-1)

    def forward(self, _input):
        self.update_distribution(_input)
        return self.distribution.sample()

class Decoder(nn.Module):
    def __init__(self, input_dim, latent_dim, mid_dims, activation="elu"):
        super(Decoder, self).__init__()

        activation = resolve_nn_activation(activation)

        layers = []

        _input = latent_dim
        for idx, dim in enumerate(mid_dims):
            _output = dim
            layers.append(nn.Linear(_input, _output))
            layers.append(activation)

            _input = _output

        _output = input_dim
        layers.append(nn.Linear(_input, _output))
        self.decoder = nn.Sequential(*layers)


    def forward(self, _input):
        return self.decoder(_input)


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, mid_dims, activation="elu"):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim, mid_dims, activation)
        self.decoder = Decoder(input_dim, latent_dim, mid_dims, activation)

    def forward(self, _input):
        return self.encoder(_input)

    @property
    def mean(self):
        return self.encoder.mean

    @property
    def std(self):
        return self.encoder.std

    @property
    def entropy(self):
        return self.encoder.entropy

    def decod(self, _input):
        return self.decoder(_input)
