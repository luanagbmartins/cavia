import math
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from arguments import parse_args
from policies.policy import Policy, weight_init


def _product_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared

class ContextEncoder(Policy, nn.Module):
    def __init__(self, input_size, output_size, latent_dim, device, hidden_sizes=(),
                 nonlinearity=F.relu, init_std=1.0, min_std=1e-6, kl_lambda=0.01):

        super(ContextEncoder, self).__init__(input_size, output_size)
        self.latent_dim = latent_dim
        self.use_ib = True
        self.kl_lambda = kl_lambda
        self.device = device

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.num_layers = len(hidden_sizes) + 1

        layer_sizes = (input_size,) + hidden_sizes
        for i in range(1, self.num_layers):
            self.add_module('layer{0}'.format(i),
                            nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
        self.output_layer = nn.Linear(layer_sizes[-1], output_size)

        self.apply(weight_init)

        # initialize buffers for z dist and z
        # use buffers so latent context can be saved along with model weights
        self.register_buffer('z', torch.zeros(1, latent_dim))
        self.register_buffer('z_means', torch.zeros(1, latent_dim))
        self.register_buffer('z_vars', torch.zeros(1, latent_dim))

        self.clear_z()

    def clear_z(self, num_tasks=1):
        """Reset q(z|c) to the prior sample a new z from the prior."""

        # reset distribution over z to the prior
        mu = torch.zeros(num_tasks, self.latent_dim).to(self.device)
        
        if self.use_ib:
            var = torch.ones(num_tasks, self.latent_dim).to(self.device)
        else:
            var = torch.zeros(num_tasks, self.latent_dim).to(self.device)
        
        self.z_means = mu
        self.z_vars = var

        # sample a new z from the prior
        self.sample_z()

        # reset the context collected so far
        self.context = None

        return torch.unbind(self.z)[0]

    def update_context(self, inputs):
        """ Append single transition to the current context."""

        o, a, r = inputs
        r = np.reshape(r, (r.shape[0], 1))

        o = torch.from_numpy(o[None, None, ...]).float().to(self.device)
        a = torch.from_numpy(a[None, None, ...]).float().to(self.device)
        r = torch.from_numpy(r[None, None, ...]).float().to(self.device)
        
        data = torch.cat([o, a, r], dim=3)

        if self.context is None:
            self.context = data
        else:
            self.context = torch.cat([self.context, data], dim=1)

    def sample_z(self):
        if self.use_ib:
            posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
            z = [d.rsample() for d in posteriors]
            self.z = torch.stack(z)
        else:
            self.z = self.z_means

    def compute_kl_div(self):
        """ compute KL( q(z|c) || r(z) ) """

        prior = torch.distributions.Normal(torch.zeros(self.latent_dim).to(self.device), torch.ones(self.latent_dim).to(self.device))
        posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
        kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
        kl_div_sum = torch.sum(torch.stack(kl_divs))
        
        return kl_div_sum

    def forward(self, input, params=None):
        """ given context, get statistics under the current policy of a set of observations """

        if params is None:
            params = OrderedDict(self.named_parameters())
        
        output = input
        for i in range(1, self.num_layers):
            output = F.linear(output,
                              weight=params['layer{0}.weight'.format(i)],
                              bias=params['layer{0}.bias'.format(i)])
            output = self.nonlinearity(output)

        output = self.output_layer(output)

        # infer_posterior
        params = output.view(input.size(0), -1, self.output_size)

        # with probabilistic z, predict mean and variance of q(z | c)
        if self.use_ib:
            mu = params[..., :self.latent_dim]
            sigma_squared = F.softplus(params[..., self.latent_dim:])
            z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
            self.z_means = torch.stack([p[0] for p in z_params])
            self.z_vars = torch.stack([p[1] for p in z_params])
        # sum rather than product of gaussians structure
        else:
            self.z_means = torch.mean(params, dim=1)
            
        self.sample_z()

        return torch.unbind(self.z)[0]
