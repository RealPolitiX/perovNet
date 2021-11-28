#! /usr/bin/env python
# -*- coding: utf-8 -*-

## Model training-related methods

import numpy as np
import parsetta as ps
import math, time
import torch


######################
# Loss functions
######################

loss_fn = torch.nn.MSELoss()
loss_fn_mae = torch.nn.L1Loss()


######################
# Training loop
######################

def loglinspace(rate, step, end=None):
    t = 0
    while end is None or t <= end:
        yield t
        t = int(t + 1 + step * (1 - math.exp(-t * rate / step)))


def evaluate(model, dataloader, device, n_norm):

    ndata = len(dataloader)
    model.eval()
    loss_cumulative = 0.
    loss_cumulative_mae = 0.
    start_time = time.time()
    
    with torch.no_grad():
        for j, d in enumerate(dataloader):
            d.to(device)
            output = model(d.x, d.edge_index, d.edge_attr, n_norm=n_norm, batch=d.batch)
            loss = loss_fn(output, d.y).cpu()
            loss_mae = loss_fn_mae(output, d.y).cpu()
            loss_cumulative = loss_cumulative + loss.detach().item()
            loss_cumulative_mae = loss_cumulative_mae + loss_mae.detach().item()
    
    return loss_cumulative / ndata, loss_cumulative_mae / ndata


def train(model, optimizer, dataloader, dataloader_valid, scheduler, n_norm, max_iter=101., device="cpu"):
    model.to(device)
    
    checkpoint_generator = loglinspace(0.3, 5)
    checkpoint = next(checkpoint_generator)
    start_time = time.time()
    dynamics = []
    
    for step in range(max_iter):
        model.train()
        loss_cumulative = 0.
        loss_cumulative_mae = 0.
        
        for j, d in enumerate(dataloader):
            d.to(device)
            output = model(d.x, d.edge_index, d.edge_attr, n_norm=n_norm, batch=d.batch)
            loss = loss_fn(output, d.y).cpu()
            loss_mae = loss_fn_mae(output, d.y).cpu()
            print(f"Iteration {step+1:4d}    batch {j+1:5d} / {len(dataloader):5d}   " +
                  f"batch loss = {loss.data}", end="\r", flush=True)
            loss_cumulative = loss_cumulative + loss.detach().item()
            loss_cumulative_mae = loss_cumulative_mae + loss_mae.detach().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        end_time = time.time()
        wall = end_time - start_time
        
        if step == checkpoint:
            checkpoint = next(checkpoint_generator)
            assert checkpoint > step
            
            valid_avg_loss = evaluate(model, dataloader_valid, device, n_norm)
            # This is the more correct thing to do 
            # -- but since evaluation takes long, we will skip it and use during batch values.
            train_avg_loss = evaluate(model, dataloader, device, n_norm)

            dynamics.append({
                'step': step,
                'wall': wall,
                'batch': {
                    'loss': loss.item(),
                    'mean_abs': loss_mae.item(),
                },
                'valid': {
                    'loss': valid_avg_loss[0],
                    'mean_abs': valid_avg_loss[1],
                },
#                 'train': {
#                     'loss': loss_cumulative / len(dataloader),
#                     'mean_abs': loss_cumulative_mae / len(dataloader),
#                 },
                'train': {
                    'loss': train_avg_loss[0],
                    'mean_abs': train_avg_loss[1],
                },
            })

            yield {
                'dynamics': dynamics,
                'state': model.state_dict()
            }
            
            print(f"Iteration {step+1:4d}    batch {j+1:5d} / {len(dataloader):5d}   " +
                  f"train loss = {train_avg_loss[0]:8.3f}   " +
                  f"valid loss = {valid_avg_loss[0]:8.3f}   " +
                  f"elapsed time = {time.strftime('%H:%M:%S', time.gmtime(wall))}")
        
        scheduler.step()


######################
# Data spllitter
######################

def split(data, portions, seed=None):
    """ Split data into different portions for training and validation.

    **Parameters**\n
        data: iterable
            Data to split.
        portions: list/tuple
            Proportions for splitting data into.
        seed: int | None
            Seed for random number generation.
    """
    
    n = len(data)
    indices = np.array(list(range(n)))
    if seed is not None:
        torch.manual_seed(seed)

    portion_indices = []
    portion_data = []
    nports = [0] + [np.rint(p*n).astype('int') for p in portions]
    np.random.shuffle(indices)
    
    for p in range(len(portions)):
        selector = slice(nports[p], nports[p] + nports[p+1])
        portion_indices.append(indices[selector])

        dats = data[selector]
        portion_data.append(dats)

    return portion_data, portion_indices