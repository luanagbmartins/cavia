import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from policies.policy import Policy, weight_init


class NormalMLPPolicy(Policy, nn.Module):
    """CAVIA network based on a multi-layer perceptron (MLP), with a
    `Normal` distribution output, with trainable standard deviation. This
    policy network can be used on tasks with continuous action spaces (eg.
    `HalfCheetahDir`).
    """

    def __init__( self, input_size, output_size, device, hidden_sizes=(), 
                  num_context_params=10, nonlinearity=F.relu, init_std=1.0, min_std=1e-6 ):
        super(NormalMLPPolicy, self).__init__(input_size, output_size)
        self.input_size = input_size
        self.output_size = output_size

        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.min_log_std = math.log(min_std)
        self.num_layers = len(hidden_sizes) + 1
        self.context_params = []

        layer_sizes = (input_size,) + hidden_sizes
        self.add_module('layer{0}'.format(1), nn.Linear(layer_sizes[0] + num_context_params, layer_sizes[1]))
        for i in range(2, self.num_layers):
            self.add_module('layer{0}'.format(i), nn.Linear(layer_sizes[i - 1], layer_sizes[i]))

        self.num_context_params = num_context_params
        self.context_params = torch.zeros(self.num_context_params, requires_grad=True).to(device)

        self.mu = nn.Linear(layer_sizes[-1], output_size)
        self.sigma = nn.Parameter(torch.Tensor(output_size))
        self.sigma.data.fill_(math.log(init_std))
        self.apply(weight_init)

    def forward(self, input):

        params = OrderedDict(self.named_parameters())
        
        # concatenate context parameters to input
        output = torch.cat((input, self.context_params.expand(input.shape[:-1] + self.context_params.shape)),
                           dim=len(input.shape) - 1)

        # forward through FC Layer
        for i in range(1, self.num_layers):
            output = F.linear(output, weight=params['layer{0}.weight'.format(i)],
                              bias=params['layer{0}.bias'.format(i)])
            output = self.nonlinearity(output)

        # last layer outputs mean; scale is a learned param independent of the input
        mu = F.linear(output, weight=params['mu.weight'], bias=params['mu.bias'])
        scale = torch.exp(torch.clamp(params['sigma'], min=self.min_log_std))

        return Normal(loc=mu, scale=scale)
