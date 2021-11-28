#! /usr/bin/env python
# -*- coding: utf-8 -*-

## Methods for crystal structure featurizations

import parsetta as ps
import torch
import pymatgen
from pymatgen.core.periodic_table import Element
import e3nn
from e3nn.point.data_helpers import DataPeriodicNeighbors


elem_dict = ps.elements.get_element_dict('mass_number')
elemnames = list(elem_dict.keys())
nelems = len(elemnames)

# Load parameters
params = ps.utils.read_json(r'../resources/params/perovnet_20210215.json')


def one_hot_elements(name, weight='mass'):
    """ One-hot encoding for chemical elements with selected weights.
    """
    
    indelem = elemnames.index(name)
    vec = torch.zeros(nelems, 1)
    vec[indelem] = 1
    
    if weight == 'mass':
        vec[indelem] *= elem_dict[name]
    else:
        try:
            # Use a weight dictionary
            vec[indelem] *= weight[name]
        except:
            vec[indelem] *= weight
        
    return vec
    

def perovskite_atom_tensor():
    """ Define a peroskite atomic type tensor (on demand).
    """
    
    A_input = torch.tensor([1., 0., 0.]).unsqueeze(0)
    B_input = torch.tensor([0., 1., 0.]).unsqueeze(0)
    X_input = torch.tensor([0., 0., 1.]).unsqueeze(0)
    
    return [A_input, B_input, X_input]


def perovskite_order_param(Rs_order_param=None):
    """ Define an order parameter for peroskite.
    """

    if Rs_order_param is None:
        Rs_order_param = [(1, 0, 1), (1, 0, -1), (1, 1, 1), (1, 1, -1), (1, 2, 1), (1, 2, -1)]
    Rs_in = [(3, 0, 1)] + Rs_order_param

    return Rs_in


def create_features(order_param_input, atom_inputs, atom_indices, zeros):
    """ Stitch together fixed atom type inputs and the learnable order parameters,
    ported and adapted from Tess Smidt's code.
    """

    order_param = torch.cat([zeros, order_param_input], dim=0)

    # N = len(atom_indices)
    all_atom_types = torch.cat([
        atom_inputs[i] for i in atom_indices
    ], dim=0)  # [N, atom_types]

    return torch.cat([all_atom_types, order_param], dim=-1)


def data_constructor(structures, properties, n_norm=40, n_elems=118, verbose=False, encode_comp='onehot_mass', **kwds):
    """ Featurize structure and property data using the tensor representation.
    """
    
    dataset = kwds.pop('dataset', [])
    for i, struct in enumerate(structures):
        print(f"Encoding sample {i+1:5d}/{len(structures):5d}", end="\r", flush=True)
        
        if encode_comp == 'onehot_mass':
            # One-hot encoding with atomic mass
            input = torch.zeros(len(struct), n_elems)
            for j, site in enumerate(struct):
                elemZ = int(Element(str(site.specie)).Z)
                input[j, elemZ] = elem.atomic_mass
        elif encode_comp == 'onehot_pettifor':
            # One-hot encoding with the Pettifor scale
            input = torch.zeros(len(struct), n_elems)
            for j, site in enumerate(struct):
                elemZ = int(Element(str(site.specie)).Z)
                input[j, elemZ] = elem.mendeleev_no
        elif encode_comp == 'onehot_mpettifor':
            mPett = ps.utils.read_json(r'../resources/params/modified_Pettifor.json')
            input = torch.zeros(len(struct), n_elems)
            for j, site in enumerate(struct):
                elemZ = int(Element(str(site.specie)).Z)
                input[j, elemZ] = mPett[str(site.specie)]
        else:
            input = encode_comp[i]
        
        # Append to existing dataset
        mat_repr = DataPeriodicNeighbors(x=input, Rs_in=None, pos=torch.tensor(struct.cart_coords.copy()),
                                         lattice=torch.tensor(struct.lattice.matrix.copy()),
                                         r_max=params['max_radius'], y=properties[i].unsqueeze(0),
                                         n_norm=n_norm)
        dataset.append(mat_repr)
    
    return dataset