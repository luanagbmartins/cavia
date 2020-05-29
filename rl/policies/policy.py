from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)


def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()


class Policy(nn.Module):
    def __init__(self, input_size, output_size):
        super(Policy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

    def update_params(self, loss, step_size, first_order=False, params=None):
        """Apply one step of gradient descent on the loss function `loss`, with 
        step-size `step_size`, and returns the updated parameters of the neural 
        network.
        """

        grads = torch.autograd.grad(loss, params, create_graph=not first_order)
        grads = parameters_to_vector(grads)
        old_params = parameters_to_vector(self.parameters())
        vector_to_parameters(old_params - step_size * grads, self.parameters())

    def reset_context(self):
        pass
