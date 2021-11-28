#! /usr/bin/env python
# -*- coding: utf-8 -*-

## The neural network models

import torch
import torch_scatter


class PerovNetShallow(torch.nn.Module):
    """ Shallow version of the perovskite neural network.
    """
    
    def __init__(self, atom_type_in, atom_type_out, model):

        super().__init__()
        self.linear = torch.nn.Linear(atom_type_in, atom_type_out)
        self.model = model
        self.relu = torch.nn.ReLU()
    
    def forward(self, x, *args, batch=None, **kwargs):
        """ Forward pass of the model.
        """

        output = self.linear(x)
        output = self.relu(output)
        output = self.model(output, *args, **kwargs)
        
        if batch is None:
            N = output.shape[0]
            batch = output.new_ones(N)
        
        output = torch_scatter.scatter_add(output, batch, dim=0)
        output = self.relu(output)
        maxima, _ = torch.max(output,axis=1)
        output = output.div(maxima.unsqueeze(1))
        
        return output