#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#     MolPROP fuses molecular language and graph for property prediction.
#     Copyright © 2023 Merck & Co., Inc., Rahway, NJ, USA and its affiliates. All rights reserved.

#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""MolPROP utilities """
from torch import nn
import torch
import torch_geometric
import math
from torch.optim.lr_scheduler import LambdaLR
from kornia.losses.focal import FocalLoss, BinaryFocalLossWithLogits
import csv
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from ray import tune
import sys
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

def getDataSetDirectories(setup, data_dir):
    """
    function collects and  returns a list of paths to directories, one for every dataset.
    Arguments:
        data_dir: str, path/dir of the dataset
    Returns:
        dataset_dirs: list of str, dataset directories
    """
    
    atomsets = setup["paths"]["use_atomsets"]
    print(atomsets)
    if atomsets is not False or None:
        atomsets = atomsets.split(', ')
    else: 
        raise ValueError("atomsets must be specified in setup file by directory name")
        
    dataset_dirs = []
    for channel_dir in atomsets:
        path = os.path.join(data_dir, channel_dir)
        for dataset_name in os.listdir(path):
           dataset_dirs.append(os.path.join(path, dataset_name))
    return dataset_dirs

class collateFunction():
    """  pad batches of data with variable number of structures. """

    def __init__(self, setup, set_L=32):
        """
        Args:
            set_L: int, pad length
        """
        self.setup = setup
        self.set_L = set_L

    def __call__(self, batch):
        return self.collateFunction(batch)

    def collateFunction(self, batch):
        """
        collate function to adjust for a variable number of structures.
        Args:
            batch: lists of structural ensemble datasets
        Returns:
            padded_lr_batch: tensor (B, C, set_L, W, H), low resolution ensembles
            padded_ens_attention_mask_batch: tensor (B, C, set_L, W, H), boolean attention mask for low resolution ensembles
            alpha_batch: tensor (B, set_L), boolean low resolution ensemble indicator (0 if padded, 1 if genuine)
            tokens_batch: tensor (B, 162), tokenized light chain sequence for ablang (padded '-, 21' to 160 with start '<, 0' and end '>, 22' tokens)
            attention_masks_batch: tensor (B, 162), boolean light chain mask for ablang 
            prop_batch: tensor (B, 1), antibody property value
            isn_batch: list of ensemble names
        """

        lr_batch = []  # batch of low resolution structures
        ens_attention_mask_batch = [] # batch of attention masks for ensemble of structures
        alpha_batch = []  # batch of indicators (0 if padded, 1 if genuine)
        tokens_batch = [] # batch of light chain sequences
        attention_masks_batch = [] # batch of light chain attention masks
        prop_batch = [] # batch of protein properties
        isn_batch = []  # batch of names

        train_batch = True

        
        for dataset in batch:
            lrs = dataset['lr']
            ens_attention_masks = dataset['ens_attention_mask']
            tokens = dataset['token']
            attention_masks = dataset['attention_mask']
            
            if type(dataset['lr']) == torch.Tensor:
                if lrs is not None:
                    C, L, H, W = lrs.shape
                else:
                    L = None

                if L is not None and L >= self.set_L: 
                    # append batch up to set_L by removing 'L's to set_L
                    ## use ensemble structures up to set_L index
                    lr_batch.append(lrs[:, :self.set_L, :, :])
                    ens_attention_mask_batch.append(ens_attention_masks[:self.set_L])
                    #ens_attention_mask_batch.append(ens_attention_masks[:, :self.set_L, : ,:])
                    alpha_batch.append(torch.ones(self.set_L))
                    ## append sequence tokens & masks for structures
                    tokens_batch.append(tokens)
                    attention_masks_batch.append(attention_masks)

                elif L is not None and L < self.set_L : # this may not work
                    # append batch up to set_L by padding 'L's to set_L
                    ## 0 pad ensemble structure distance maps to set_L
                    pad = torch.full((C, self.set_L - L, H, W), fill_value=0)
                    lr_batch.append(torch.cat([lrs, pad], dim=1))
                    ens_attention_mask_batch.append(torch.cat([ens_attention_masks, pad], dim=1))
                    alpha_batch.append(torch.cat([torch.ones(L), torch.zeros(self.set_L - L)], dim=0))
                    ## append sequence tokens & masks for structures
                    tokens_batch.append(tokens)
                    attention_masks_batch.append(attention_masks)
                else:
                    lr_batch = None
                    ens_attention_mask_batch = None
                    alpha_batch = None
                    tokens_batch.append(tokens)
                    attention_masks_batch.append(attention_masks)

                
                # append batch of properties to predict
                prop = dataset['prop']
                if train_batch and prop is not None:
                    prop_batch.append(prop)
                else:
                    train_batch = False


                isn_batch.append(dataset['name'])

            elif type(dataset['lr']) == torch_geometric.data.data.Data:
            
                L = None
                lr_batch.append(lrs)
                #padded_lr_batch = lr_batch
                #padded_lr_batch = torch.stack(lr_batch, dim=0)
                ens_attention_mask_batch = None
                padded_ens_attention_mask_batch = None
                alpha_batch = None
                tokens_batch.append(tokens)
                attention_masks_batch.append(attention_masks)
                prop = dataset['prop']
                if train_batch and prop is not None:
                    prop_batch.append(prop)
                else:
                    train_batch = False


                isn_batch.append(dataset['name'])
            
            elif dataset['lr'] is None:
                L = None
                ens_attention_mask_batch = None
                padded_ens_attention_mask_batch = None
                alpha_batch = None
                tokens_batch.append(tokens)
                attention_masks_batch.append(attention_masks)
                prop = dataset['prop']
                if train_batch and prop is not None:
                    prop_batch.append(prop)
                else:
                    train_batch = False


                isn_batch.append(dataset['name'])

         # convert lists batch to torch stacks batch
        if L is not None:
            padded_lr_batch = torch.stack(lr_batch, dim=0)
            padded_ens_attention_mask_batch = torch.stack(ens_attention_mask_batch, dim=0)
            alpha_batch = torch.stack(alpha_batch, dim=0)
        elif L is None and type(dataset['lr']) == torch_geometric.data.data.Data:
            from torch_geometric.data import Batch
            #padded_lr_batch = Batch.from_data_list(lr_batch)
            padded_lr_batch = lr_batch
            padded_ens_attention_mask_batch = None
            alpha_batch = None
        else:
            padded_lr_batch = None
            padded_ens_attention_mask_batch = None
            alpha_batch = None
            
        if tokens is not None:
            tokens_batch = torch.stack(tokens_batch, dim=0).squeeze(1)
            attention_masks_batch = torch.stack(attention_masks_batch, dim=0).squeeze(1)
        else:
            tokens_batch = None
            attention_masks_batch = None

        if train_batch:
            prop_batch = torch.stack(prop_batch, dim=0)
        
        return padded_lr_batch, padded_ens_attention_mask_batch, alpha_batch, tokens_batch, attention_masks_batch, prop_batch, isn_batch

def get_loss(prop_preds, props, metric='L2', num_predictions=1):
    """
    compute loss for batch instance.
    Args:
        prop_preds: tensor (B, 1), batch property predictions from ALEF
        props: tensor (B, 1), batch properties
    Returns:
        loss: tensor (B), loss metric for antibody property prediction.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    props = props.to(device)
    prop_preds = prop_preds.to(device)
    
    if metric == 'L1':
        loss = torch.nn.L1Loss().to(device)
        l1s = loss(props, prop_preds)
        return l1s

    if metric == 'L2':
        loss = torch.nn.MSELoss().to(device)
        l2s = loss(props, prop_preds)
        return l2s
    
    if metric == 'L3':
        loss = torch.nn.MSELoss().to(device)
        l3s = torch.sqrt(loss(props, prop_preds))
        return l3s
    
    if metric == 'focal':
        if num_predictions > 2:
            # not sure if this works for multi-class but close enough for now
            alpha = torch.tensor([1/num_predictions]*num_predictions)
            alpha = alpha.float()
            props = props.to(torch.int64)
            loss = FocalLoss(alpha = alpha, gamma = 2.0, reduction = 'mean').to(device)
            focal_loss = loss(prop_preds, props)
        elif num_predictions == 1:
            alpha = 0.5
            props = props.to(torch.int64)
            loss = BinaryFocalLossWithLogits(alpha = alpha, gamma = 2.0, reduction = 'mean').to(device)
            focal_loss = loss(prop_preds, props)
            return focal_loss
    if metric == 'BCE':
        loss = torch.nn.BCEWithLogitsLoss().to(device)
        focal_loss = loss(prop_preds, props)
        return focal_loss



def get_inverse_sqrt_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that linearly increases from 0 until the configured
    learning rates (lr_coder, lr_transformer, lr_lang). Thereafter we decay proportional to the inverse square root of
    the number of epochs. Scheduler implemented in BERT.

    Sources:
    https://github.com/jcyk/copyisallyouneed/blob/master/optim.py
    """

    def lr_lambda(current_step):
        current_step = max(1, current_step)
        return min(float(current_step)**-0.5, float(current_step)*(float(num_warmup_steps)**-1.5))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)

