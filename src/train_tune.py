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

""" script to train and tune MolPROP for small molecule property prediction """
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from scipy import stats
import json
import os
from os.path import join
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
from os.path import join, exists, dirname, basename, isfile
from collections import defaultdict
import glob
import argparse
from itertools import chain
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import torch_geometric
from torch.optim import lr_scheduler
from torch.utils.data import Dataset,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from DeepNetworks.SLEF import SLEFNet
from DataLoader import SuperSet, SuperSet_csv, DataSet
from utils import getDataSetDirectories, collateFunction, get_loss, get_inverse_sqrt_schedule_with_warmup
from tensorboardX import SummaryWriter
from  torch.cuda.amp import autocast
import ray
from ray import air
from ray import tune
from ray.air import session
from ray.tune import CLIReporter
from ray.air.checkpoint import Checkpoint
from ray.tune.stopper import ExperimentPlateauStopper
from ray.tune.schedulers import HyperBandForBOHB, ASHAScheduler
from ray.tune.search.bohb import TuneBOHB


def train_model(fusion_model, optimizer, lr_function, dataloaders, setup):
    #with autocast(enabled=False):
    # seed for reproducibility
    eval('setattr(torch.backends.cudnn, "deterministic", True)')
    eval('setattr(torch.backends.cudnn, "benchmark", True)')
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    np.random.seed(setup["training"]["seed"])  
    torch.manual_seed(setup["training"]["seed"])
    #device = torch.device('cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # redefine num_epochs for ray tune implementation
    if setup["training"]["train_all_data"]:
        num_epochs = setup["training"]["num_epochs"]
        min_loss = setup["training"]["min_loss"]
        patience_flag = 0
    else:
        num_epochs = 1 # epochs are set in ray tune if hyperparameter tuning

    # load pretrained model
    if setup["training"]["load_checkpoints"]:
            if len(glob.glob(join(checkpoint_dir_run, f'SLEF.pth'))) > 0:
                print("loading fusion model from training checkpoint")
                module_state_dict = torch.load(os.path.join(checkpoint_dir_run, 
                                                    'SLEF.pth'))
                optim_state_dict = module_state_dict["optimizer_state_dict"]
                optim_state_dict = {k[7:]: v for k, v in optim_state_dict.items()}
                module_state_dict = module_state_dict["model_state_dict"]
                model_state_dict = {k[7:]: v for k, v in module_state_dict.items()}
                fusion_model.load_state_dict(model_state_dict)
                optimizer.load_state_dict(optim_state_dict) 

            elif len(glob.glob(join(checkpoint_dir_run, f'SLEF_validation.pth'))) > 0:
                print("loading fusion model from validation checkpoint")
                module_state_dict = torch.load(os.path.join(checkpoint_dir_run, 
                                                    'SLEF_validation.pth'))
                optim_state_dict = module_state_dict["optimizer_state_dict"]
                optim_state_dict = {k[7:]: v for k, v in optim_state_dict.items()}
                module_state_dict = module_state_dict["model_state_dict"]
                model_state_dict = {k[7:]: v for k, v in module_state_dict.items()}
                fusion_model.load_state_dict(model_state_dict)
                optimizer.load_state_dict(optim_state_dict) 
            else:
                pass
    

    
    #sys.exit()
    for epoch in tqdm(range(1, num_epochs + 1)):

        # torch train mode
        fusion_model.train()
        train_loss = 0.0

        # iterate over data batches
        for lrs, ens_attention_masks, alphas, tokens, attention_masks, props, names in tqdm(dataloaders["train"]):
            optimizer.zero_grad() # zero the parameter gradients
            if type(lrs) is torch.Tensor or lrs is None:
                if setup["paths"]["precision"] == "16":
                    lrs = lrs.half().to(device)
                    ens_attention_masks = ens_attention_masks.bool().to(device)
                    alphas = alphas.half().to(device)
                    tokens = tokens.int().to(device)
                    attention_masks.bool().to(device)
                    props = props.half().to(device)
                elif setup["paths"]["precision"] == "32":
                    lrs = lrs.float().to(device)
                    ens_attention_masks = ens_attention_masks.bool().to(device)
                    alphas = alphas.bool().to(device)
                    tokens = tokens.int().to(device)
                    attention_masks.bool().to(device)
                    props = props.float().to(device)
                elif setup["paths"]["precision"] == "64":
                    lrs = lrs.double().to(device)
                    ens_attention_masks = ens_attention_masks.bool().to(device)
                    alphas = alphas.bool().to(device)
                    tokens = tokens.int().to(device)
                    attention_masks.bool().to(device)
                    props = props.double().to(device)
            else:
                lrs = [lr.to(device) for lr in lrs]
                from torch_geometric.data import Batch
                lrs = Batch.from_data_list(lrs).to(device)
                tokens = tokens.int().to(device)
                attention_masks.bool().to(device)
                props = props.float().to(device)

            fusion_size = 200

            prop_preds = fusion_model(lrs, ens_attention_masks, alphas, tokens, attention_masks, fusion_size=fusion_size) 
            prop_preds = prop_preds.to(device)

            # training loss
            if setup['training']['combine_losses']:
                loss1 = get_loss(prop_preds, props, metric='L1')
                loss2 = get_loss(prop_preds, props, metric='L2')
                loss = setup['training']['alpha1'] * loss1 + setup['training']['alpha2'] * loss2
            else:
                loss = get_loss(prop_preds, props, metric=setup['training']['loss'], num_predictions=setup['network']['language']['num_predictions'])
            
            loss = torch.mean(loss)

            # backpropogate loss through network
            loss.backward()
            optimizer.step()
            
            if setup['training']["kfold"]:
                batch_loss = loss.detach().cpu().numpy() * len(props) / len(dataloaders['train'].sampler.indices)
            else:
                batch_loss = loss.detach().cpu().numpy() * len(props) / len(dataloaders['train'].dataset)
            
            train_loss += batch_loss

            if setup["training"]["verbose"]:
                print('')
                print('batch_loss: ', loss.detach().cpu().numpy())
                print('train_loss: ', train_loss)
                print('')

        if setup["training"]["train_all_data"]:
            if epoch % 100 == 0:
                torch.save({'model_state_dict': fusion_model.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(), 
                    'train_loss': train_loss}, 
                    os.path.join(checkpoint_dir_run, 'SLEF_%s.pth' % epoch))
                
        lr_function.step()

        if setup["training"]["train_all_data"]:
            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('train/lr_ens_coder', optimizer.state_dict()['param_groups'][0]['lr'], epoch)
            writer.add_scalar('train/lr_ens_transformer', optimizer.state_dict()['param_groups'][1]['lr'], epoch)
            writer.add_scalar('train/lr_lang', optimizer.state_dict()['param_groups'][2]['lr'], epoch)
            print('epoch: ', epoch, 'patience_flag: ', patience_flag,
            ' lr_coder: ', optimizer.state_dict()['param_groups'][0]['lr'],
            ' lr_transformer: ', optimizer.state_dict()['param_groups'][1]['lr'],
            ' lr_lang: ', optimizer.state_dict()['param_groups'][2]['lr'])

        if setup["training"]["train_all_data"]:
            if train_loss <= min_loss:
                min_loss = train_loss
                patience_flag = 0
                # save model
                torch.save({'model_state_dict': fusion_model.state_dict(), 
                        'optimizer_state_dict': optimizer.state_dict(), 
                        'train_loss': train_loss}, 
                        os.path.join(checkpoint_dir_run, 'SLEF.pth'))
            else:
                patience_flag += 1
                print('')
        print('')
        print('train_loss: ', train_loss)
        print('')

    del lrs, ens_attention_masks, alphas, tokens, attention_masks, props, names
    return train_loss
            

def validate_model(fusion_model, optimizer, dataloaders, setup):
    #device = torch.device('cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #torch evaluation mode
    fusion_model.eval()
    PROPs = []
    PREDs = []
    val_score = 0.0

    for lrs, ens_attention_masks, alphas, tokens, attention_masks, props, names in tqdm(dataloaders["val"]):
        if type(lrs) is torch.Tensor or lrs is None:
            if setup["paths"]["precision"] == "16":
                lrs = lrs.half().to(device)
                ens_attention_masks = ens_attention_masks.bool().to(device)
                alphas = alphas.half().to(device)
                tokens = tokens.int().to(device)
                attention_masks.bool().to(device)
                props = props.half().to(device)
            elif setup["paths"]["precision"] == "32":
                lrs = lrs.float().to(device)
                ens_attention_masks = ens_attention_masks.bool().to(device)
                alphas = alphas.bool().to(device)
                tokens = tokens.int().to(device)
                attention_masks.bool().to(device)
                props = props.float().to(device)
            elif setup["paths"]["precision"] == "64":
                lrs = lrs.double().to(device)
                ens_attention_masks = ens_attention_masks.bool().to(device)
                alphas = alphas.bool().to(device)
                tokens = tokens.int().to(device)
                attention_masks.bool().to(device)
                props = props.double().to(device)
        else:                
            lrs = [lr.to(device) for lr in lrs]
            from torch_geometric.data import Batch
            lrs = Batch.from_data_list(lrs).to(device)
            tokens = tokens.int().to(device)
            attention_masks.bool().to(device)
            props = props.float().to(device)
        
        fusion_size = 320
            
        prop_preds = fusion_model(lrs, ens_attention_masks, alphas, tokens, attention_masks, fusion_size=fusion_size)
        score = 0
        for i in range(prop_preds.shape[0]):
            PROPs.append(props[i].detach().cpu().numpy())  
            PREDs.append(prop_preds[i].detach().cpu().numpy())
        
        if setup['training']['combine_losses']:
            score1 = get_loss(prop_preds, props, metric='L1')
            score2 = get_loss(prop_preds, props, metric='L2')
            score = setup['training']['alpha1'] * score1 + setup['training']['alpha2'] * score2
        else:
            score = get_loss(prop_preds,props, metric=setup['training']['loss'], num_predictions=setup['network']['language']['num_predictions'])
        
        score = torch.mean(score)
        
        if setup['training']["kfold"]:
            batch_score = score.detach().cpu().numpy() * len(props) / len(dataloaders['val'].sampler.indices)
        else:
            batch_score = score.detach().cpu().numpy() * len(props) / len(dataloaders['val'].dataset)
        val_score += batch_score

    PROPs = np.squeeze(np.array(PROPs))
    PREDs = np.squeeze(np.array(PREDs))

    if setup['network']['language']['mode'] == 'continuous':
        
        r2 = r2_score(PROPs, PREDs)
        pearson = stats.pearsonr(PROPs, PREDs).statistic
        spearman = stats.spearmanr(PROPs, PREDs).correlation

        print('')
        print('r2: ', r2, 'rp: ', pearson, 'rs: ', spearman)
        print('val_score:  ', val_score)
        print('')
        del lrs, ens_attention_masks, alphas, tokens, attention_masks, props, names
        return val_score, r2, pearson, spearman
    
    elif setup['network']['language']['mode'] == 'discrete':
        if  setup['network']['language']['num_predictions'] == 1:
            # convert logits
            PREDs = torch.from_numpy(PREDs)
            PREDs = PREDs.unsqueeze(1)
            PREDs = torch.sigmoid(PREDs)
            PREDs = PREDs.squeeze()
            PREDs = PREDs.detach().cpu().numpy()
            PREDs = np.round(PREDs)
            acc = accuracy_score(PROPs, PREDs)
            ap = average_precision_score(PROPs, PREDs)
            auc = roc_auc_score(PROPs, PREDs)
            print('')
            print('acc: ', acc, 'prc: ', ap, 'roc_auc: ', auc)
            print('val_score:  ', val_score)
            print('')
        elif setup['network']['language']['num_predictions'] > 2:
            PREDs = PREDs.argmax(axis=1)
            acc = None
            ap = average_precision_score(PROPs, PREDs)
            auc = roc_auc_score(PROPs, PREDs)
            print('')
            print('prc: ', ap, 'roc_auc: ', auc)
            print('val_score:  ', val_score)
            print('')
        del lrs, ens_attention_masks, alphas, tokens, attention_masks, props, names
        return val_score, acc, ap, auc

def hp_tune(config, setup, checkpoint_dir=None):
    os.chdir(cwd)
    # random seeds for reproducibility
    np.random.seed(setup["training"]["seed"]) 
    torch.manual_seed(setup["training"]["seed"])     
    # define compute specifications
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    auto_garbage_collect()
    torch.cuda.empty_cache()
    
    if setup["training"]["val_proportion"]:
        dataloaders = data_import()
        fusion_model = SLEFNet(config["network"])
        fusion_model = torch.nn.DataParallel(fusion_model, device_ids=list(range(torch.cuda.device_count()))).to(device)

        if config["network"]["language"]["graph"] == False:
            # language model parameters and learning rates
            if config["network"]["language"]["model"] == 'chemberta-77m-mlm' or config["network"]["language"]["model"] == 'chemberta-10m-mlm' or config["network"]["language"]["model"] == 'chemberta-77m-mtr' or config["network"]["language"]["model"] == 'chemberta-10m-mtr':
                transformer_params = list(map(id, fusion_model.module.superres.parameters()))
                coder_params = filter(lambda p: id(p) not in transformer_params, fusion_model.module.parameters())
                chemberta_params = filter(lambda p: id(p) in coder_params, fusion_model.module.property.chemberta.parameters())
                params = [
                    {"params": coder_params, "lr": config["learning_rates"]["lr_ens_coder"]},
                    {"params": fusion_model.module.superres.parameters(), "lr": config["learning_rates"]["lr_ens_transformer"]},
                    {"params": chemberta_params, "lr": config["learning_rates"]["lr_lang"]}
                ]
            elif config["network"]["language"]["model"] == 'chemberta-77m-mlm-only' or config["network"]["language"]["model"] == 'chemberta-10m-mlm-only' or config["network"]["language"]["model"] == 'chemberta-77m-mtr-only' or config["network"]["language"]["model"] == 'chemberta-10m-mtr-only':
                coder_params = fusion_model.module.parameters()
                chemberta_params = filter(lambda p: id(p) in coder_params, fusion_model.module.property.chemberta.parameters())
                params = [
                    {"params": coder_params, "lr": config["learning_rates"]["lr_ens_coder"]},
                    {"params": chemberta_params, "lr": config["learning_rates"]["lr_lang"]}
                ]
            elif config["network"]["language"]["model"] == 'none':
                transformer_params = list(map(id, fusion_model.module.superres.parameters()))
                coder_params = filter(lambda p: id(p) not in transformer_params, fusion_model.module.parameters())
                params = [
                    {"params": coder_params, "lr": config["learning_rates"]["lr_ens_coder"]},
                    {"params": fusion_model.module.superres.parameters(), "lr": config["learning_rates"]["lr_ens_transformer"]}
                ]
        elif config["network"]["language"]["graph"] == "GCN":
            if config["network"]["language"]["model"] == 'chemberta-77m-mlm' or config["network"]["language"]["model"] == 'chemberta-10m-mlm' or config["network"]["language"]["model"] == 'chemberta-77m-mtr' or config["network"]["language"]["model"] == 'chemberta-10m-mtr':
                all_params = fusion_model.module.GCNproperty.parameters()
                chem_params = list(map(id, fusion_model.module.GCNproperty.chemberta.parameters()))
                graph_params = filter(lambda p: id(p) not in chem_params, all_params)
                chemberta_params = filter(lambda p: id(p) in all_params, fusion_model.module.GCNproperty.chemberta.parameters())
                params = [
                    {"params": graph_params, "lr": config["learning_rates"]["lr_ens_coder"]},
                    {"params": chemberta_params, "lr": config["learning_rates"]["lr_lang"]}
                ]
            elif config["network"]["language"]["model"] == 'chemberta-77m-mlm-only' or config["network"]["language"]["model"] == 'chemberta-10m-mlm-only' or config["network"]["language"]["model"] == 'chemberta-77m-mtr-only' or config["network"]["language"]["model"] == 'chemberta-10m-mtr-only' or config["network"]["language"]["model"] == 'none':
                raise ValueError("graph models must be used with language model")
        elif config["network"]["language"]["graph"] == "GAT":
            if config["network"]["language"]["model"] == 'chemberta-77m-mlm' or config["network"]["language"]["model"] == 'chemberta-10m-mlm' or config["network"]["language"]["model"] == 'chemberta-77m-mtr' or config["network"]["language"]["model"] == 'chemberta-10m-mtr':
                all_params = fusion_model.module.GATproperty.parameters()
                chem_params = list(map(id, fusion_model.module.GATproperty.chemberta.parameters()))
                graph_params = filter(lambda p: id(p) not in chem_params, all_params)
                chemberta_params = filter(lambda p: id(p) in all_params, fusion_model.module.GATproperty.chemberta.parameters())
                params = [
                    {"params": graph_params, "lr": config["learning_rates"]["lr_ens_coder"]},
                    {"params": chemberta_params, "lr": config["learning_rates"]["lr_lang"]}
                ]
            elif config["network"]["language"]["model"] == 'chemberta-77m-mlm-only' or config["network"]["language"]["model"] == 'chemberta-10m-mlm-only' or config["network"]["language"]["model"] == 'chemberta-77m-mtr-only' or config["network"]["language"]["model"] == 'chemberta-10m-mtr-only' or config["network"]["language"]["model"] == 'none':
                raise ValueError("graph models must be used with language model")
            
        elif config["network"]["language"]["graph"] == "GATv2":
            if config["network"]["language"]["model"] == 'chemberta-77m-mlm' or config["network"]["language"]["model"] == 'chemberta-10m-mlm' or config["network"]["language"]["model"] == 'chemberta-77m-mtr' or config["network"]["language"]["model"] == 'chemberta-10m-mtr':
                all_params = fusion_model.module.GATv2property.parameters()
                chem_params = list(map(id, fusion_model.module.GATv2property.chemberta.parameters()))
                graph_params = filter(lambda p: id(p) not in chem_params, all_params)
                chemberta_params = filter(lambda p: id(p) in all_params, fusion_model.module.GATv2property.chemberta.parameters())
                params = [
                    {"params": graph_params, "lr": config["learning_rates"]["lr_ens_coder"]},
                    {"params": chemberta_params, "lr": config["learning_rates"]["lr_lang"]}
                ]
            elif config["network"]["language"]["model"] == 'chemberta-77m-mlm-only' or config["network"]["language"]["model"] == 'chemberta-10m-mlm-only' or config["network"]["language"]["model"] == 'chemberta-77m-mtr-only' or config["network"]["language"]["model"] == 'chemberta-10m-mtr-only' or config["network"]["language"]["model"] == 'none':
                raise ValueError("graph models must be used with language model")
            
        optimizer = optim.Adam(params)

        if config["learning_rates"]["lr_strategy"] == 0:
            lr_function = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config["learning_rates"]['lr_decay'],
                                                    verbose=True, patience=config["learning_rates"]['lr_step'])
        elif config["learning_rates"]["lr_strategy"] == 1:
            lr_function = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["learning_rates"]["num_epochs"], 
                                                    eta_min=optimizer.param_groups[1]['lr']/100.)
        elif config["learning_rates"]["lr_strategy"] == 2:
            lr_function = lr_scheduler.StepLR(optimizer, step_size=config["learning_rates"]['lr_step'], 
                                                    gamma=config["learning_rates"]['lr_decay'])
        elif config["learning_rates"]["lr_strategy"] == 3:
            lr_function = get_inverse_sqrt_schedule_with_warmup(optimizer, config['learning_rates']['lr_step'], setup['training']['num_epochs'])
        
        epochs = config.get("epochs", setup["training"]["num_epochs"])
        start = 0
        loaded_checkpoint = session.get_checkpoint()
        if loaded_checkpoint:
            checkpoint_dict = loaded_checkpoint.to_dict()
            fusion_model.load_state_dict(checkpoint_dict.get("model_state_dict"))
            last_step = loaded_checkpoint.to_dict()["step"]
            start = last_step + 1   

        for step in range(start, epochs):
            while True:
                train_score = train_model(fusion_model, optimizer, lr_function, dataloaders, setup)
                if not setup['training']['loss'] == 'focal':
                    val_score, r2, pearson, spearman = validate_model(fusion_model, optimizer, dataloaders, setup)
                    metrics = {"train_loss": float(train_score), "validation_loss": float(val_score), "r2": float(r2), "rp": float(pearson), "rs": float(spearman)}
                elif setup['training']['loss'] == 'focal':
                    val_score, acc, ap, auc = validate_model(fusion_model, optimizer, dataloaders, setup)
                    metrics = {"train_loss": float(train_score), "validation_loss": float(val_score), "prc": float(ap), "auc": float(auc)}
                state_dict = fusion_model.state_dict()
                #consume_prefix_in_state_dict_if_present(state_dict, "module.") # NOT for DataParallel
                checkpoint = Checkpoint.from_dict({"step": step, 'model_state_dict': state_dict, 
                        'validation_loss': val_score})
                session.report(metrics, checkpoint=checkpoint)
                torch.cuda.empty_cache()

    elif setup["training"]["kfold"]:

        torch.cuda.empty_cache()
        data_directory = setup["paths"]["data_directory"]
        random_seed = setup["training"]["seed"]
        k_count = setup['training']['kfold']
        batch_size = setup["training"]["batch_size"]
        n_workers = setup["training"]["n_workers"]
        ens_L = setup["training"]["ens_L"]
        set_L = setup["training"]["set_L"]
        chemprop = setup["paths"]["chemprop_maps"]
        median_ref = setup["paths"]["median_ref"]
        lang = config["network"]["language"]["model"]
        graph = config["network"]["language"]["graph"]
        num_predictions = config["network"]["language"]["num_predictions"]

        foldperf = {}
        splits = KFold(n_splits=k_count, shuffle=True, random_state=random_seed)
        
        # construct data objects
        if graph:
            train = pd.read_csv(os.path.join(data_directory, setup["training"]["dataset"]))
            train_dataset = SuperSet_csv(data = train, X = 'ids' , y = 'y')
        else:
            train_list = getDataSetDirectories(setup, os.path.join(data_directory, "train"))
            train_dataset = SuperSet(superset_dir=train_list,
                                    setup=setup["training"],
                                    ens_L=ens_L, lang=lang, graph=graph, chemprop=chemprop, median_ref=median_ref)
        
        epochs = config.get("epochs", setup["training"]["num_epochs"])
        start = 0
        loaded_checkpoint = session.get_checkpoint()
        if loaded_checkpoint:
            last_step = loaded_checkpoint.to_dict()["step"]
            start = last_step + 1  
        
        for step in range(start, epochs):
            while True:
                model_state_dicts = {'step': step, 'fold': [] , 'model_state_dict': [], 'optimizer_state_dict':[], 'scheduler_state_dict':[]}
                if setup['network']['language']['mode'] == 'continuous':
                    cv_data = {'train_loss': [], 'val_loss': [],'r2':[],'rp':[],'rs':[]}
                elif setup['network']['language']['mode'] == 'discrete':
                    cv_data = {'train_loss': [], 'val_loss': [],'prc':[],'roc_auc':[]}
                print(' ')
                for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(train_dataset)))):
                    print('fold: %s' % str(fold + 1)  + ' ' + 'train_len: %s' % str(len(train_idx)) + ' ' +'val_len: %s' % str(len(val_idx)))
                print(' ')
                for fold, (train_idx, val_idx) in tqdm(enumerate(splits.split(np.arange(len(train_dataset)))), total = int(k_count)):
                    torch.cuda.empty_cache()
                    fusion_model = SLEFNet(config["network"])
                    fusion_model = torch.nn.DataParallel(fusion_model, device_ids=list(range(torch.cuda.device_count()))).to(device)
                    # language model parameters and learning rates
                    if config["network"]["language"]["graph"] == False:
                        # language model parameters and learning rates
                        if config["network"]["language"]["model"] == 'chemberta-77m-mlm' or config["network"]["language"]["model"] == 'chemberta-10m-mlm' or config["network"]["language"]["model"] == 'chemberta-77m-mtr' or config["network"]["language"]["model"] == 'chemberta-10m-mtr':
                            transformer_params = list(map(id, fusion_model.module.superres.parameters()))
                            coder_params = filter(lambda p: id(p) not in transformer_params, fusion_model.module.parameters())
                            chemberta_params = filter(lambda p: id(p) in coder_params, fusion_model.module.property.chemberta.parameters())
                            params = [
                                {"params": coder_params, "lr": config["learning_rates"]["lr_ens_coder"]},
                                {"params": fusion_model.module.superres.parameters(), "lr": config["learning_rates"]["lr_ens_transformer"]},
                                {"params": chemberta_params, "lr": config["learning_rates"]["lr_lang"]}
                            ]
                        elif config["network"]["language"]["model"] == 'chemberta-77m-mlm-only' or config["network"]["language"]["model"] == 'chemberta-10m-mlm-only' or config["network"]["language"]["model"] == 'chemberta-77m-mtr-only' or config["network"]["language"]["model"] == 'chemberta-10m-mtr-only':
                            coder_params = fusion_model.module.parameters()
                            chemberta_params = filter(lambda p: id(p) in coder_params, fusion_model.module.property.chemberta.parameters())
                            params = [
                                {"params": coder_params, "lr": config["learning_rates"]["lr_ens_coder"]},
                                {"params": chemberta_params, "lr": config["learning_rates"]["lr_lang"]}
                            ]
                        elif config["network"]["language"]["model"] == 'none':
                            transformer_params = list(map(id, fusion_model.module.superres.parameters()))
                            coder_params = filter(lambda p: id(p) not in transformer_params, fusion_model.module.parameters())
                            params = [
                                {"params": coder_params, "lr": config["learning_rates"]["lr_ens_coder"]},
                                {"params": fusion_model.module.superres.parameters(), "lr": config["learning_rates"]["lr_ens_transformer"]}
                            ]
                    elif config["network"]["language"]["graph"] == "GCN":
                        if config["network"]["language"]["model"] == 'chemberta-77m-mlm' or config["network"]["language"]["model"] == 'chemberta-10m-mlm' or config["network"]["language"]["model"] == 'chemberta-77m-mtr' or config["network"]["language"]["model"] == 'chemberta-10m-mtr':
                            all_params = fusion_model.module.GCNproperty.parameters()
                            chem_params = list(map(id, fusion_model.module.GCNproperty.chemberta.parameters()))
                            graph_params = filter(lambda p: id(p) not in chem_params, all_params)
                            chemberta_params = filter(lambda p: id(p) in all_params, fusion_model.module.GCNproperty.chemberta.parameters())
                            params = [
                                {"params": graph_params, "lr": config["learning_rates"]["lr_ens_coder"]},
                                {"params": chemberta_params, "lr": config["learning_rates"]["lr_lang"]}
                            ]
                        elif config["network"]["language"]["model"] == 'chemberta-77m-mlm-only' or config["network"]["language"]["model"] == 'chemberta-10m-mlm-only' or config["network"]["language"]["model"] == 'chemberta-77m-mtr-only' or config["network"]["language"]["model"] == 'chemberta-10m-mtr-only' or config["network"]["language"]["model"] == 'none':
                            raise ValueError("graph models must be used with language model")
                    elif config["network"]["language"]["graph"] == "GAT":
                        if config["network"]["language"]["model"] == 'chemberta-77m-mlm' or config["network"]["language"]["model"] == 'chemberta-10m-mlm' or config["network"]["language"]["model"] == 'chemberta-77m-mtr' or config["network"]["language"]["model"] == 'chemberta-10m-mtr':
                            all_params = fusion_model.module.GATproperty.parameters()
                            chem_params = list(map(id, fusion_model.module.GATproperty.chemberta.parameters()))
                            graph_params = filter(lambda p: id(p) not in chem_params, all_params)
                            chemberta_params = filter(lambda p: id(p) in all_params, fusion_model.module.GATproperty.chemberta.parameters())
                            params = [
                                {"params": graph_params, "lr": config["learning_rates"]["lr_ens_coder"]},
                                {"params": chemberta_params, "lr": config["learning_rates"]["lr_lang"]}
                            ]
                        elif config["network"]["language"]["model"] == 'chemberta-77m-mlm-only' or config["network"]["language"]["model"] == 'chemberta-10m-mlm-only' or config["network"]["language"]["model"] == 'chemberta-77m-mtr-only' or config["network"]["language"]["model"] == 'chemberta-10m-mtr-only' or config["network"]["language"]["model"] == 'none':
                            raise ValueError("graph models must be used with language model")
                        
                    elif config["network"]["language"]["graph"] == "GATv2":
                        if config["network"]["language"]["model"] == 'chemberta-77m-mlm' or config["network"]["language"]["model"] == 'chemberta-10m-mlm' or config["network"]["language"]["model"] == 'chemberta-77m-mtr' or config["network"]["language"]["model"] == 'chemberta-10m-mtr':
                            all_params = fusion_model.module.GATv2property.parameters()
                            chem_params = list(map(id, fusion_model.module.GATv2property.chemberta.parameters()))
                            graph_params = filter(lambda p: id(p) not in chem_params, all_params)
                            chemberta_params = filter(lambda p: id(p) in all_params, fusion_model.module.GATv2property.chemberta.parameters())
                            params = [
                                {"params": graph_params, "lr": config["learning_rates"]["lr_ens_coder"]},
                                {"params": chemberta_params, "lr": config["learning_rates"]["lr_lang"]}
                            ]
                        elif config["network"]["language"]["model"] == 'chemberta-77m-mlm-only' or config["network"]["language"]["model"] == 'chemberta-10m-mlm-only' or config["network"]["language"]["model"] == 'chemberta-77m-mtr-only' or config["network"]["language"]["model"] == 'chemberta-10m-mtr-only' or config["network"]["language"]["model"] == 'none':
                            raise ValueError("graph models must be used with language model")

                    optimizer = optim.Adam(params)

                    if config["learning_rates"]["lr_strategy"] == 0:
                        lr_function = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config["learning_rates"]['lr_decay'],
                                                                verbose=True, patience=config["learning_rates"]['lr_step'])
                    elif config["learning_rates"]["lr_strategy"] == 1:
                        lr_function = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["learning_rates"]["num_epochs"], 
                                                                eta_min=optimizer.param_groups[1]['lr']/100.)
                    elif config["learning_rates"]["lr_strategy"] == 2:
                        lr_function = lr_scheduler.StepLR(optimizer, step_size=config["learning_rates"]['lr_step'], 
                                                                gamma=config["learning_rates"]['lr_decay'])
                    elif config["learning_rates"]["lr_strategy"] == 3:
                        lr_function = get_inverse_sqrt_schedule_with_warmup(optimizer, config['learning_rates']['lr_step'], setup['training']['num_epochs'])
                    f = str(fold+1)
                    print('')
                    print('fold {}'.format(fold + 1))
                    print('')
                    loaded_checkpoint = session.get_checkpoint()
                    if loaded_checkpoint:
                        checkpoint_dict = loaded_checkpoint.to_dict()
                        fold_dict = checkpoint_dict.get("model_state_dict")
                        fusion_model.load_state_dict(fold_dict[fold])
                        lr_dict = checkpoint_dict.get("scheduler_state_dict")
                        lr_function.load_state_dict(lr_dict[fold])
                        optim_dict = checkpoint_dict.get("optimizer_state_dict")
                        optimizer.load_state_dict(optim_dict[fold])
                        #for state in optimizer.state.values():
                            #for k, v in state.items():
                                #if isinstance(v, torch.Tensor):
                                    #state[k] = v.cuda()
                    train_sampler = SubsetRandomSampler(train_idx)
                    test_sampler = SubsetRandomSampler(val_idx)
                    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=n_workers,
                                            collate_fn=collateFunction(setup=setup, set_L=set_L), drop_last=True,
                                            pin_memory=True)
                    val_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=n_workers,
                                            collate_fn=collateFunction(setup=setup, set_L=set_L), drop_last=True,
                                            pin_memory=True)
                    dataloaders = {'train': train_dataloader, 'val': val_dataloader}
                    train_score = train_model(fusion_model, optimizer, lr_function, dataloaders, setup)

                    if setup['network']['language']['mode'] == 'continuous':
                        val_score, r2, pearson, spearman = validate_model(fusion_model, optimizer, dataloaders, setup)
                        cv_data['train_loss'].append(train_score)
                        cv_data['val_loss'].append(val_score)
                        cv_data['r2'].append(r2)
                        cv_data['rp'].append(pearson)
                        cv_data['rs'].append(spearman)
                        model_state_dicts['fold'].append(f)
                        state_dict = fusion_model.state_dict()
                        state_dict = {k: v.cpu() for k, v in state_dict.items()}
                        model_state_dicts['model_state_dict'].append(state_dict)
                        lr_state_dict = lr_function.state_dict()
                        model_state_dicts['scheduler_state_dict'].append(lr_state_dict)
                        for state in optimizer.state.values():
                            for k, v in state.items():
                                if isinstance(v, torch.Tensor):
                                    state[k] = v.cpu()
                        optimizer_state_dict = optimizer.state_dict()
                        model_state_dicts['optimizer_state_dict'].append(optimizer_state_dict)
                    elif setup['network']['language']['mode'] == 'discrete':
                        val_score, acc, ap, auc = validate_model(fusion_model, optimizer, dataloaders, setup)
                        cv_data['train_loss'].append(train_score)
                        cv_data['val_loss'].append(val_score)
                        cv_data['prc'].append(ap)
                        cv_data['roc_auc'].append(auc)
                        model_state_dicts['fold'].append(f)
                        state_dict = fusion_model.state_dict()
                        state_dict = {k: v.cpu() for k, v in state_dict.items()}
                        model_state_dicts['model_state_dict'].append(state_dict)
                        lr_state_dict = lr_function.state_dict()
                        model_state_dicts['scheduler_state_dict'].append(lr_state_dict)
                        for state in optimizer.state.values():
                            for k, v in state.items():
                                if isinstance(v, torch.Tensor):
                                    state[k] = v.cpu()
                        optimizer_state_dict = optimizer.state_dict()   
                        model_state_dicts['optimizer_state_dict'].append(optimizer_state_dict)
                    del fusion_model
                    del optimizer
                    del state_dict
                    del lr_state_dict
                    del optimizer_state_dict
                    auto_garbage_collect()
                
                if setup['network']['language']['mode'] == 'continuous':

                    train_loss_min_std = np.std(cv_data["train_loss"])
                    train_loss_min = np.mean(cv_data["train_loss"])
                    val_loss_min_std = np.std(cv_data["val_loss"])
                    val_loss_min = np.mean(cv_data["val_loss"])
                    r2_max_std = np.std(cv_data["r2"])
                    r2_max = np.mean(cv_data["r2"])
                    rp_max_std = np.std(cv_data["rp"])
                    rp_max = np.mean(cv_data["rp"])
                    rs_max_std = np.std(cv_data["rs"])
                    rs_max = np.mean(cv_data["rs"])

                    print('')
                    print('train_loss_avg: %s' %train_loss_min)
                    print('train_loss_std: %s' %train_loss_min_std)
                    print('val_loss_avg: %s' %val_loss_min)
                    print('val_loss_std: %s' %val_loss_min_std)
                    print('r2_avg: %s' %r2_max)
                    print('r2_std: %s' %r2_max_std)
                    print('rp_avg: %s' %rp_max)
                    print('rp_std: %s' %rp_max_std)
                    print('rs_avg: %s' %rs_max)
                    print('rs_std: %s' %rs_max_std)
                    print('')
                    metrics = {"train_loss": float(train_loss_min), "validation_loss": float(val_loss_min), "r2": float(r2_max), "rp": float(rp_max), "rs": float(rs_max)}
                elif setup['network']['language']['mode'] == 'discrete':
                    train_loss_min_std = np.std(cv_data["train_loss"])
                    train_loss_min = np.mean(cv_data["train_loss"])
                    val_loss_min_std = np.std(cv_data["val_loss"])
                    val_loss_min = np.mean(cv_data["val_loss"])
                    prc_max_std = np.std(cv_data["prc"])
                    prc_max = np.mean(cv_data["prc"])
                    auc_max_std = np.std(cv_data["roc_auc"])
                    auc_max = np.mean(cv_data["roc_auc"])

                    print('')
                    print('train_loss_avg: %s' %train_loss_min)
                    print('train_loss_std: %s' %train_loss_min_std)
                    print('val_loss_avg: %s' %val_loss_min)
                    print('val_loss_std: %s' %val_loss_min_std)
                    print('prc_avg: %s' %prc_max)
                    print('prc_std: %s' %prc_max_std)
                    print('roc_auc_avg: %s' %auc_max)
                    print('roc_auc_std: %s' %auc_max_std)
                    print('')
                    metrics = {"train_loss": float(train_loss_min), "validation_loss": float(val_loss_min), "prc": float(prc_max), "roc_auc": float(auc_max)}


                #state_dict = fusion_model.state_dict()
                #consume_prefix_in_state_dict_if_present(state_dict, "module.") # NOT for DataParallel
                checkpoint = Checkpoint.from_dict(model_state_dicts)
                session.report(metrics, checkpoint=checkpoint)
                del model_state_dicts
                del cv_data
                del checkpoint
                del metrics
                torch.cuda.empty_cache()
                auto_garbage_collect()
                
            #print(next(iter(dataloaders['train'])))
            

        
                    
def data_import():

    # collect lists of directory paths which contain structural ensembles

    data_directory = setup["paths"]["data_directory"]
    random_seed = setup["training"]["seed"]

    if setup["training"]["train_all_data"] and not setup["network"]["language"]["graph"]:
        train_list = getDataSetDirectories(setup, os.path.join(data_directory, "train"))
        
    elif not setup["network"]["language"]["graph"]:
        train_set_directories = getDataSetDirectories(setup, os.path.join(data_directory, "train"))

        # organize all directoroes by atomset, split into train and val, and 
        molecule_dict_list = defaultdict(list)
        for dir_ in train_set_directories:
            molecule_dict_list[basename(dir_)].append(dir_)
        molecule_list = [[k] for k, v in molecule_dict_list.items()]

        t_list, v_list = train_test_split(molecule_list,
        test_size=setup['training']['val_proportion'],
        random_state=random_seed, shuffle=True)

        t_list = list(chain.from_iterable(t_list))
        v_list = list(chain.from_iterable(v_list))
        train_list, val_list = [], []
        for dir_ in train_set_directories:
            if basename(dir_) in t_list:
                train_list.append(dir_)
            elif basename(dir_) in v_list:
                val_list.append(dir_)
            else: 
                raise ValueError("molecule not found in training or validation set")
    elif setup["network"]["language"]["graph"]:
        pass

    # Dataloaders
    batch_size = setup["training"]["batch_size"]
    n_workers = setup["training"]["n_workers"]
    ens_L = setup["training"]["ens_L"]
    set_L = setup["training"]["set_L"]
    lang = setup["network"]["language"]["model"]
    graph = setup["network"]["language"]["graph"]
    chemprop = setup["paths"]["chemprop_maps"]
    median_ref = setup["paths"]["median_ref"]

    if not graph:
        train_dataset = SuperSet(superset_dir=train_list,
                                        setup=setup["training"],
                                        ens_L=ens_L, 
                                        lang=lang,
                                        graph = graph, 
                                        chemprop=chemprop, 
                                        median_ref=median_ref)
        
        train_dataloader = DataLoader(train_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=n_workers,
                                    collate_fn=collateFunction(setup=setup, set_L=set_L),
                                    pin_memory=True)

        if setup["training"]["train_all_data"]:
            dataloaders = {'train': train_dataloader}
        else:
            val_dataset = SuperSet(superset_dir=val_list,
                                        setup=setup["training"],
                                        ens_L=ens_L, 
                                        lang=lang, 
                                        graph=graph,
                                        chemprop=chemprop, 
                                        median_ref=median_ref)
            val_dataloader = DataLoader(val_dataset,
                                        batch_size=int(round(batch_size/3)),
                                        shuffle=False,
                                        num_workers=n_workers,
                                        collate_fn=collateFunction(setup=setup, set_L=set_L),
                                        pin_memory=True)
            dataloaders = {'train': train_dataloader, 'val': val_dataloader}
    else:
        if setup["training"]["train_all_data"]:
            if setup["training"]["dataset"]:
                train = pd.read_csv(os.path.join(data_directory, setup["training"]["dataset"]))
                train_dataset = SuperSet_csv(data = train, X = 'ids' , y = 'y')
            else:
                raise ValueError("dataset not found in specified directory: %s" % str(os.path.join(data_directory, setup["training"]["dataset"])))
            train_dataloader = DataLoader(train_dataset,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=n_workers,
                                        collate_fn=collateFunction(setup=setup, set_L=set_L),
                                        pin_memory=True)
            dataloaders = {'train': train_dataloader}
        else:
            if setup["training"]["dataset"]:
                #combine string data directory and csv file
                data = pd.read_csv(os.path.join(data_directory, setup["training"]["dataset"]))
                train = data.sample(frac=setup['training']['val_proportion'], random_state=random_seed).reset_index(drop=True)
                valid = data.loc[~data.index.isin(train.index)].reset_index(drop=True)
                train_dataset = SuperSet_csv(data = train, X = 'ids' , y = 'y')
                valid_dataset = SuperSet_csv(data = valid, X = 'ids' , y = 'y')
            else:
                raise ValueError("dataset not found in specified directory: %s" % str(os.path.join(data_directory, setup["training"]["dataset"])))
            
            train_dataloader = DataLoader(train_dataset,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=n_workers,
                                        collate_fn=collateFunction(setup=setup, set_L=set_L),
                                        pin_memory=True)
            val_dataloader = DataLoader(valid_dataset,
                                        batch_size=int(round(batch_size/3)),
                                        shuffle=False,
                                        num_workers=n_workers,
                                        collate_fn=collateFunction(setup=setup, set_L=set_L),
                                        pin_memory=True)
            dataloaders = {'train': train_dataloader, 'val': val_dataloader}
    torch.cuda.empty_cache()
    return dataloaders

def auto_garbage_collect(pct=70.0):
    """
    auto_garbage_collection - Call the garbage collection if memory used is greater than 70% of total available memory.
                              This is called to deal with an issue in Ray not freeing up used memory.

        pct - Default value of 70%.  Amount of memory in use that triggers the garbage collection call.
    """
    import gc
    import psutil
    if psutil.virtual_memory().percent >= pct:
        gc.collect()
    return

class Logger():
    def __init__(self, logging_dir=None):
        self.terminal = sys.stdout
        self.log = open(logging_dir + '/train_tune.log', 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        pass
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", help="path of the setup file", default='config/setup.json')
    args = parser.parse_args()
    assert os.path.isfile(args.setup)

    with open(args.setup, "r") as read_file:
        setup = json.load(read_file)
        
    cwd = os.getcwd()
    

    batch_size = setup["training"]["batch_size"]
    ensl = setup["training"]['ens_L']
    subfolder_pattern = 'batch_{}_time_{}'.format(batch_size, f"{datetime.datetime.now():%Y-%m-%d-%H-%M-%S-%f}")

    print('')
    print('ens_L: %s' % str(ensl))
    print('saved: %s' % str(subfolder_pattern))

    checkpoint_dir_run = os.path.join(setup["paths"]["checkpoint_dir"], subfolder_pattern)
    os.makedirs(checkpoint_dir_run, exist_ok=True)

    #tensor board logging if train_all_data (ray tune tensorboard data in logging_dir/ray_tune)
    tb_logging_dir = setup['paths']['log_dir']
    logging_dir = os.path.join(tb_logging_dir, subfolder_pattern)
    os.makedirs(logging_dir, exist_ok=True)

    # write console output to log file
    sys.stdout = Logger(logging_dir)
    save_setup = pd.read_json(args.setup)
    save_setup.to_json(os.path.join(checkpoint_dir_run, 'setup.json'))
    save_setup.to_json(os.path.join(logging_dir, 'setup.json'))

    if setup["training"]["ray_tune"] == False:

        writer = SummaryWriter(logging_dir)

        # random seeds for reproducibility
        np.random.seed(setup["training"]["seed"]) 
        torch.manual_seed(setup["training"]["seed"])
        #device = torch.device('cpu')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        if setup["training"]["train_all_data"] or setup["training"]["val_proportion"]:
            dataloaders = data_import()

            fusion_model = SLEFNet(setup["network"])
            fusion_model = torch.nn.DataParallel(fusion_model, device_ids=list(range(torch.cuda.device_count()))).to(device)

            # language model parameters and learning rates
            if setup["network"]["language"]["graph"] == False:
                if setup["network"]["language"]["model"] == 'chemberta-77m-mlm' or setup["network"]["language"]["model"] == 'chemberta-10m-mlm' or setup["network"]["language"]["model"] == 'chemberta-77m-mtr' or setup["network"]["language"]["model"] == 'chemberta-10m-mtr':
                    transformer_params = list(map(id, fusion_model.module.superres.parameters()))
                    coder_params = filter(lambda p: id(p) not in transformer_params, fusion_model.module.parameters())
                    chemberta_params = filter(lambda p: id(p) in coder_params, fusion_model.module.property.chemberta.parameters())
                    params = [
                        {"params": coder_params, "lr": setup["training"]["lr_coder"]},
                        {"params": fusion_model.module.superres.parameters(), "lr": setup["training"]["lr_transformer"]},
                        {"params": chemberta_params, "lr": setup["training"]["lr_lang"]}
                    ]
                elif setup["network"]["language"]["model"] == 'chemberta-77m-mlm-only' or setup["network"]["language"]["model"] == 'chemberta-10m-mlm-only' or setup["network"]["language"]["model"] == 'chemberta-77m-mtr-only' or setup["network"]["language"]["model"] == 'chemberta-10m-mtr-only':
                    coder_params = fusion_model.module.parameters()
                    chemberta_params = filter(lambda p: id(p) in coder_params, fusion_model.module.property.chemberta.parameters())
                    params = [
                        {"params": coder_params, "lr": setup["training"]["lr_coder"]},
                        {"params": chemberta_params, "lr": setup["training"]["lr_lang"]}
                    ]
                elif setup["network"]["language"]["model"] == 'none':
                    transformer_params = list(map(id, fusion_model.module.superres.parameters()))
                    coder_params = filter(lambda p: id(p) not in transformer_params, fusion_model.module.parameters())
                    params = [
                        {"params": coder_params, "lr": setup["training"]["lr_coder"]},
                        {"params": fusion_model.module.superres.parameters(), "lr": setup["training"]["lr_transformer"]}
                    ]
            elif setup["network"]["language"]["graph"] == "GCN":
                if setup["network"]["language"]["model"] == 'chemberta-77m-mlm' or setup["network"]["language"]["model"] == 'chemberta-10m-mlm' or setup["network"]["language"]["model"] == 'chemberta-77m-mtr' or setup["network"]["language"]["model"] == 'chemberta-10m-mtr':
                    all_params = fusion_model.module.GCNproperty.parameters()
                    chem_params = list(map(id, fusion_model.module.GCNproperty.chemberta.parameters()))
                    graph_params = filter(lambda p: id(p) not in chem_params, all_params)
                    chemberta_params = filter(lambda p: id(p) in all_params, fusion_model.module.GCNproperty.chemberta.parameters())
                    params = [
                        {"params": graph_params, "lr": setup["training"]["lr_coder"]},
                        {"params": chemberta_params, "lr": setup["training"]["lr_lang"]}
                    ]
                elif setup["network"]["language"]["model"] == 'chemberta-77m-mlm-only' or setup["network"]["language"]["model"] == 'chemberta-10m-mlm-only' or setup["network"]["language"]["model"] == 'chemberta-77m-mtr-only' or setup["network"]["language"]["model"] == 'chemberta-10m-mtr-only' or setup["network"]["language"]["model"] == 'none':
                    raise ValueError("graph models must be used with language model")
            elif setup["network"]["language"]["graph"] == "GAT":
                if setup["network"]["language"]["model"] == 'chemberta-77m-mlm' or setup["network"]["language"]["model"] == 'chemberta-10m-mlm' or setup["network"]["language"]["model"] == 'chemberta-77m-mtr' or setup["network"]["language"]["model"] == 'chemberta-10m-mtr':
                    all_params = fusion_model.module.GATproperty.parameters()
                    chem_params = list(map(id, fusion_model.module.GATproperty.chemberta.parameters()))
                    graph_params = filter(lambda p: id(p) not in chem_params, all_params)
                    chemberta_params = filter(lambda p: id(p) in all_params, fusion_model.module.GATproperty.chemberta.parameters())
                    params = [
                        {"params": graph_params, "lr": setup["training"]["lr_coder"]},
                        {"params": chemberta_params, "lr": setup["training"]["lr_lang"]}
                    ]
                elif setup["network"]["language"]["model"] == 'chemberta-77m-mlm-only' or setup["network"]["language"]["model"] == 'chemberta-10m-mlm-only' or setup["network"]["language"]["model"] == 'chemberta-77m-mtr-only' or setup["network"]["language"]["model"] == 'chemberta-10m-mtr-only' or setup["network"]["language"]["model"] == 'none':
                    raise ValueError("graph models must be used with language model")
            
            elif setup["network"]["language"]["graph"] == "GATv2":
                if setup["network"]["language"]["model"] == 'chemberta-77m-mlm' or setup["network"]["language"]["model"] == 'chemberta-10m-mlm' or setup["network"]["language"]["model"] == 'chemberta-77m-mtr' or setup["network"]["language"]["model"] == 'chemberta-10m-mtr':
                    all_params = fusion_model.module.GATv2property.parameters()
                    chem_params = list(map(id, fusion_model.module.GATv2property.chemberta.parameters()))
                    graph_params = filter(lambda p: id(p) not in chem_params, all_params)
                    chemberta_params = filter(lambda p: id(p) in all_params, fusion_model.module.GATv2property.chemberta.parameters())
                    params = [
                        {"params": graph_params, "lr": setup["training"]["lr_coder"]},
                        {"params": chemberta_params, "lr": setup["training"]["lr_lang"]}
                    ]
                elif setup["network"]["language"]["model"] == 'chemberta-77m-mlm-only' or setup["network"]["language"]["model"] == 'chemberta-10m-mlm-only' or setup["network"]["language"]["model"] == 'chemberta-77m-mtr-only' or setup["network"]["language"]["model"] == 'chemberta-10m-mtr-only' or setup["network"]["language"]["model"] == 'none':
                    raise ValueError("graph models must be used with language model")

            optimizer = optim.Adam(params)

            if setup["training"]["lr_strategy"] == 0:
                lr_function = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=setup['training']['lr_decay'],
                                                    verbose=True, patience=setup['training']['lr_step'])
            elif setup["training"]["lr_strategy"] == 1:
                lr_function = lr_scheduler.CosineAnnealingLR(optimizer, T_max=setup["training"]["num_epochs"], 
                                                    eta_min=optimizer.param_groups[1]['lr']/100.)
            elif setup["training"]["lr_strategy"] == 2:
                lr_function = lr_scheduler.StepLR(optimizer, step_size=setup['training']['lr_step'], 
                                                    gamma=setup['training']['lr_decay'])
            elif setup["training"]["lr_strategy"] == 3:
                lr_function = get_inverse_sqrt_schedule_with_warmup(optimizer, setup['training']['lr_step'], setup['training']['num_epochs'])

        if setup["training"]["train_all_data"]:
            train_score = train_model(fusion_model, optimizer, lr_function, dataloaders, setup)
        elif setup["training"]["kfold"]:
            data_directory = setup["paths"]["data_directory"]
            random_seed = setup["training"]["seed"]
            k_count = setup['training']['kfold']
            batch_size = setup["training"]["batch_size"]
            n_workers = setup["training"]["n_workers"]
            ens_L = setup["training"]["ens_L"]
            set_L = setup["training"]["set_L"]
            lang = setup["network"]["language"]["model"]
            graph = setup["network"]["language"]["graph"]
            chemprop = setup["paths"]["chemprop_maps"]
            median_ref = setup["paths"]["median_ref"]
            foldperf = {}
            splits = KFold(n_splits=k_count, shuffle=True, random_state=random_seed)

            # obtain data
            if graph:
                train = pd.read_csv(os.path.join(data_directory, setup["training"]["dataset"]))
                train_dataset = SuperSet_csv(data = train, X = 'ids' , y = 'y')
            else:
                train_list = getDataSetDirectories(setup, os.path.join(data_directory, "train"))
                train_dataset = SuperSet(superset_dir=train_list,
                                        setup=setup["training"],
                                        ens_L=ens_L, lang=lang, graph=graph, chemprop=chemprop, median_ref=median_ref)
            
            print(' ')
            for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(train_dataset)))):
                print('fold: %s' % str(fold + 1)  + ' ' + 'train_len: %s' % str(len(train_idx)) + ' ' +'val_len: %s' % str(len(val_idx)))
            print(' ')

            for fold, (train_idx, val_idx) in tqdm(enumerate(splits.split(np.arange(len(train_dataset)))), total = k_count):
                torch.cuda.empty_cache()
                fusion_model = SLEFNet(setup["network"])
                #fusion_model = torch.nn.DataParallel(fusion_model, device_ids=[0]).to(device) 
                fusion_model = torch.nn.DataParallel(fusion_model, device_ids=list(range(torch.cuda.device_count()))).to(device)
                
                # fusion model learning rates
                
                if setup["network"]["language"]["graph"] == False:
                    if setup["network"]["language"]["model"] == 'chemberta-77m-mlm' or setup["network"]["language"]["model"] == 'chemberta-10m-mlm' or setup["network"]["language"]["model"] == 'chemberta-77m-mtr' or setup["network"]["language"]["model"] == 'chemberta-10m-mtr':
                        transformer_params = list(map(id, fusion_model.module.superres.parameters()))
                        coder_params = filter(lambda p: id(p) not in transformer_params, fusion_model.module.parameters())
                        chemberta_params = filter(lambda p: id(p) in coder_params, fusion_model.module.property.chemberta.parameters())
                        params = [
                            {"params": coder_params, "lr": setup["training"]["lr_coder"]},
                            {"params": fusion_model.module.superres.parameters(), "lr": setup["training"]["lr_transformer"]},
                            {"params": chemberta_params, "lr": setup["training"]["lr_lang"]}
                        ]
                    elif setup["network"]["language"]["model"] == 'chemberta-77m-mlm-only' or setup["network"]["language"]["model"] == 'chemberta-10m-mlm-only' or setup["network"]["language"]["model"] == 'chemberta-77m-mtr-only' or setup["network"]["language"]["model"] == 'chemberta-10m-mtr-only':
                        coder_params = fusion_model.module.parameters()
                        chemberta_params = filter(lambda p: id(p) in coder_params, fusion_model.module.property.chemberta.parameters())
                        params = [
                            {"params": coder_params, "lr": setup["training"]["lr_coder"]},
                            {"params": chemberta_params, "lr": setup["training"]["lr_lang"]}
                        ]
                    elif setup["network"]["language"]["model"] == 'none':
                        transformer_params = list(map(id, fusion_model.module.superres.parameters()))
                        coder_params = filter(lambda p: id(p) not in transformer_params, fusion_model.module.parameters())
                        params = [
                            {"params": coder_params, "lr": setup["training"]["lr_coder"]},
                            {"params": fusion_model.module.superres.parameters(), "lr": setup["training"]["lr_transformer"]}
                        ]
                elif setup["network"]["language"]["graph"] == "GCN":
                    if setup["network"]["language"]["model"] == 'chemberta-77m-mlm' or setup["network"]["language"]["model"] == 'chemberta-10m-mlm' or setup["network"]["language"]["model"] == 'chemberta-77m-mtr' or setup["network"]["language"]["model"] == 'chemberta-10m-mtr':
                        all_params = fusion_model.module.GCNproperty.parameters()
                        chem_params = list(map(id, fusion_model.module.GCNproperty.chemberta.parameters()))
                        graph_params = filter(lambda p: id(p) not in chem_params, all_params)
                        chemberta_params = filter(lambda p: id(p) in all_params, fusion_model.module.GCNproperty.chemberta.parameters())
                        params = [
                            {"params": graph_params, "lr": setup["training"]["lr_coder"]},
                            {"params": chemberta_params, "lr": setup["training"]["lr_lang"]}
                        ]
                    elif setup["network"]["language"]["model"] == 'chemberta-77m-mlm-only' or setup["network"]["language"]["model"] == 'chemberta-10m-mlm-only' or setup["network"]["language"]["model"] == 'chemberta-77m-mtr-only' or setup["network"]["language"]["model"] == 'chemberta-10m-mtr-only' or setup["network"]["language"]["model"] == 'none':
                        raise ValueError("graph models must be used with language model")
                elif setup["network"]["language"]["graph"] == "GAT":
                    if setup["network"]["language"]["model"] == 'chemberta-77m-mlm' or setup["network"]["language"]["model"] == 'chemberta-10m-mlm' or setup["network"]["language"]["model"] == 'chemberta-77m-mtr' or setup["network"]["language"]["model"] == 'chemberta-10m-mtr':
                        all_params = fusion_model.module.GATproperty.parameters()
                        chem_params = list(map(id, fusion_model.module.GATproperty.chemberta.parameters()))
                        graph_params = filter(lambda p: id(p) not in chem_params, all_params)
                        chemberta_params = filter(lambda p: id(p) in all_params, fusion_model.module.GATproperty.chemberta.parameters())
                        params = [
                            {"params": graph_params, "lr": setup["training"]["lr_coder"]},
                            {"params": chemberta_params, "lr": setup["training"]["lr_lang"]}
                        ]
                    elif setup["network"]["language"]["model"] == 'chemberta-77m-mlm-only' or setup["network"]["language"]["model"] == 'chemberta-10m-mlm-only' or setup["network"]["language"]["model"] == 'chemberta-77m-mtr-only' or setup["network"]["language"]["model"] == 'chemberta-10m-mtr-only' or setup["network"]["language"]["model"] == 'none':
                        raise ValueError("graph models must be used with language model")
                elif setup["network"]["language"]["graph"] == "GATv2":
                    if setup["network"]["language"]["model"] == 'chemberta-77m-mlm' or setup["network"]["language"]["model"] == 'chemberta-10m-mlm' or setup["network"]["language"]["model"] == 'chemberta-77m-mtr' or setup["network"]["language"]["model"] == 'chemberta-10m-mtr':
                        all_params = fusion_model.module.GATv2property.parameters()
                        chem_params = list(map(id, fusion_model.module.GATv2property.chemberta.parameters()))
                        graph_params = filter(lambda p: id(p) not in chem_params, all_params)
                        chemberta_params = filter(lambda p: id(p) in all_params, fusion_model.module.GATv2property.chemberta.parameters())
                        params = [
                            {"params": graph_params, "lr": setup["training"]["lr_coder"]},
                            {"params": chemberta_params, "lr": setup["training"]["lr_lang"]}
                        ]
                    elif setup["network"]["language"]["model"] == 'chemberta-77m-mlm-only' or setup["network"]["language"]["model"] == 'chemberta-10m-mlm-only' or setup["network"]["language"]["model"] == 'chemberta-77m-mtr-only' or setup["network"]["language"]["model"] == 'chemberta-10m-mtr-only' or setup["network"]["language"]["model"] == 'none':
                        raise ValueError("graph models must be used with language model")

                optimizer = optim.Adam(params)


                #print('---------------------GRAD_PARAMS---------------------')
                #for name, param in fusion_model.named_parameters():
                #    if param.requires_grad:
                #        print(name)
                #print('-----------------------------------------------------------')
        

                if setup["training"]["lr_strategy"] == 0:
                    lr_function = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=setup['training']['lr_decay'],
                                                        verbose=True, patience=setup['training']['lr_step'])
                elif setup["training"]["lr_strategy"] == 1:
                    lr_function = lr_scheduler.CosineAnnealingLR(optimizer, T_max=setup["training"]["num_epochs"], 
                                                        eta_min=optimizer.param_groups[1]['lr']/100.)
                elif setup["training"]["lr_strategy"] == 2:
                    lr_function = lr_scheduler.StepLR(optimizer, step_size=setup['training']['lr_step'], 
                                                        gamma=setup['training']['lr_decay'])
                elif setup["training"]["lr_strategy"] == 3:
                    lr_function = get_inverse_sqrt_schedule_with_warmup(optimizer, setup['training']['lr_step'], setup['training']['num_epochs'])

                f = str(fold+1)
                print('')
                print('fold {}'.format(fold + 1))
                print('')
                train_sampler = SubsetRandomSampler(train_idx)
                test_sampler = SubsetRandomSampler(val_idx)
                train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=n_workers,
                                        collate_fn=collateFunction(setup=setup, set_L=set_L), drop_last = True,
                                        pin_memory=True)
                val_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=n_workers,
                                        collate_fn=collateFunction(setup=setup, set_L=set_L), drop_last = True,
                                        pin_memory=True)
                dataloaders = {'train': train_dataloader, 'val': val_dataloader}
                if setup['network']['language']['mode'] == 'continuous':
                    cv_data = {'train_loss': [], 'val_loss': [],'r2':[],'rp':[],'rs':[],'lr_ens_coder':[],'lr_ens_transformer':[],'lr_lang':[]}
                elif setup['network']['language']['mode'] == 'discrete':
                    cv_data = {'train_loss': [], 'val_loss': [],'prc':[],'roc_auc':[],'lr_ens_coder':[],'lr_ens_transformer':[],'lr_lang':[]}
                num_epochs = setup["training"]["num_epochs"]
                best_val_score = setup["training"]["best_val_score"]
                min_loss = setup["training"]["min_loss"]
                patience_train = 0
                patience_val = 0
                #print(next(iter(dataloaders['train'])))

                for epoch in tqdm(range(1, num_epochs + 1)): 
                    train_score = train_model(fusion_model, optimizer, lr_function, dataloaders, setup)
                    if setup['network']['language']['mode'] == 'continuous':
                        val_score, r2, pearson, spearman = validate_model(fusion_model, optimizer, dataloaders, setup)
                        writer.add_scalar('train/f%s/loss' % f, train_score, epoch)
                        writer.add_scalar('train/f%s/patience' % f, patience_train, epoch)
                        writer.add_scalar('val/f%s/loss' % f, val_score, epoch)
                        writer.add_scalar('val/f%s/patience' % f, patience_val, epoch)
                        writer.add_scalar('val/f%s/r2' % f, r2, epoch)
                        writer.add_scalar('val/f%s/rp' % f, pearson, epoch)
                        writer.add_scalar('val/f%s/rs' % f, spearman, epoch)
                        cv_data['train_loss'].append(train_score)
                        cv_data['val_loss'].append(val_score)
                        cv_data['r2'].append(r2)
                        cv_data['rp'].append(pearson)
                        cv_data['rs'].append(spearman)
                    elif setup['network']['language']['mode'] == 'discrete':
                        val_score, acc, prc, auc = validate_model(fusion_model, optimizer, dataloaders, setup)
                        writer.add_scalar('train/f%s/loss' % f, train_score, epoch)
                        writer.add_scalar('train/f%s/patience' % f, patience_train, epoch)
                        writer.add_scalar('val/f%s/loss' % f, val_score, epoch)
                        writer.add_scalar('val/f%s/patience' % f, patience_val, epoch)
                        writer.add_scalar('val/f%s/prc' % f, prc, epoch)
                        writer.add_scalar('val/f%s/roc_auc' % f, auc, epoch)
                        cv_data['train_loss'].append(train_score)
                        cv_data['val_loss'].append(val_score)
                        cv_data['prc'].append(prc)
                        cv_data['roc_auc'].append(auc)

                    if setup["network"]["language"]["graph"]:
                        writer.add_scalar('val/lr_coder', optimizer.state_dict()['param_groups'][0]['lr'], epoch)
                        writer.add_scalar('val/lr_lang', optimizer.state_dict()['param_groups'][1]['lr'], epoch)
                    else:
                        if  setup["network"]["language"]["model"] == 'chemberta-77m-mlm' or setup["network"]["language"]["model"] == 'chemberta-10m-mlm' or setup["network"]["language"]["model"] == 'chemberta-77m-mtr' or setup["network"]["language"]["model"] == 'chemberta-10m-mtr':
                            writer.add_scalar('val/f%s/lr_ens_coder' % f, optimizer.state_dict()['param_groups'][0]['lr'], epoch)
                            writer.add_scalar('val/f%s/lr_ens_transformer' % f, optimizer.state_dict()['param_groups'][1]['lr'], epoch)
                            writer.add_scalar('val/f%s/lr_lang' % f, optimizer.state_dict()['param_groups'][2]['lr'], epoch)
                            cv_data['lr_ens_coder'].append(optimizer.state_dict()['param_groups'][0]['lr'])
                            cv_data['lr_ens_transformer'].append(optimizer.state_dict()['param_groups'][1]['lr'])
                            cv_data['lr_lang'].append(optimizer.state_dict()['param_groups'][2]['lr'])
                        elif setup["network"]["language"]["model"] == 'chemberta-77m-mlm-only' or setup["network"]["language"]["model"] == 'chemberta-10m-mlm-only' or setup["network"]["language"]["model"] == 'chemberta-77m-mtr-only' or setup["network"]["language"]["model"] == 'chemberta-10m-mtr-only':
                            writer.add_scalar('val/f%s/lr_lang' % f, optimizer.state_dict()['param_groups'][0]['lr'], epoch)
                            cv_data['lr_lang'].append(optimizer.state_dict()['param_groups'][0]['lr'])
                        elif setup["network"]["language"]["model"] == "none":
                            writer.add_scalar('val/f%s/lr_ens_coder' % f, optimizer.state_dict()['param_groups'][0]['lr'], epoch)
                            writer.add_scalar('val/f%s/lr_ens_transformer' % f, optimizer.state_dict()['param_groups'][1]['lr'], epoch)
                            cv_data['lr_ens_coder'].append(optimizer.state_dict()['param_groups'][0]['lr'])
                            cv_data['lr_ens_transformer'].append(optimizer.state_dict()['param_groups'][1]['lr'])

                    if epoch % 50 == 0:
                        torch.save({'model_state_dict': fusion_model.state_dict(), 
                        'optimizer_state_dict': optimizer.state_dict()}, 
                        os.path.join(checkpoint_dir_run, 'SLEF_f%s_%s.pth' % (f, epoch)))
                    
                    if train_score <= min_loss:
                        min_loss = train_score
                        patience_train = 0
                        torch.save({'model_state_dict': fusion_model.state_dict(), 
                                'optimizer_state_dict': optimizer.state_dict(), 
                                'train_loss': train_score}, 
                                os.path.join(checkpoint_dir_run, 'SLEF_f%s.pth' % f))
                    else:
                        patience_train += 1

                    if best_val_score > val_score:
                        torch.save({'model_state_dict': fusion_model.state_dict(), 
                                    'optimizer_state_dict': optimizer.state_dict(), 
                                    'validation_loss': val_score}, 
                        os.path.join(checkpoint_dir_run, 'SLEF_f%s_validation.pth' % f))
                        best_val_score = val_score
                        patience_val = 0
                    else:
                        patience_val += 1
                    
                    print('')
                    print('fold: %s' %f)
                    print('epoch: %s' %epoch)
                    print('min_loss: %s' %min_loss)
                    print('train_patience: %s' %patience_train)
                    print('best_val_score: %s' %best_val_score)
                    print('validation_patience: %s' %patience_val)
                    print('')
                foldperf['fold{}'.format(fold+1)] = cv_data

            if setup['network']['language']['mode'] == 'continuous':
                train_loss_min = []
                val_loss_min = []
                r2_max = []
                rp_max = []
                rs_max = []
                for fold in range(k_count):
                    fold_train_loss_min = min(foldperf['fold{}'.format(fold+1)]['train_loss'])
                    fold_val_loss_min = min(foldperf['fold{}'.format(fold+1)]['val_loss'])
                    fold_r2_max = max(foldperf['fold{}'.format(fold+1)]['r2'])
                    fold_rp_max = max(foldperf['fold{}'.format(fold+1)]['rp'])
                    fold_rs_max = max(foldperf['fold{}'.format(fold+1)]['rs'])
                    train_loss_min.append(fold_train_loss_min)
                    val_loss_min.append(fold_val_loss_min)
                    r2_max.append(fold_r2_max)
                    rp_max.append(fold_rp_max)
                    rs_max.append(fold_rs_max)
                
                train_loss_min_std = np.std(train_loss_min)
                train_loss_min = np.mean(train_loss_min)
                val_loss_min_std = np.std(val_loss_min)
                val_loss_min = np.mean(val_loss_min)
                r2_max_std = np.std(r2_max)
                r2_max = np.mean(r2_max)
                rp_max_std = np.std(rp_max)
                rp_max = np.mean(rp_max)
                rs_max_std = np.std(rs_max)
                rs_max = np.mean(rs_max)
                
                print('')
                print('train_loss_avg: %s' %train_loss_min)
                print('train_loss_std: %s' %train_loss_min_std)
                print('val_loss_avg: %s' %val_loss_min)
                print('val_loss_min_std: %s' %val_loss_min_std)
                print('r2_avg: %s' %r2_max)
                print('r2_std: %s' %r2_max_std)
                print('rp_avg: %s' %rp_max)
                print('rp_std: %s' %rp_max_std)
                print('rs_avg: %s' %rs_max)
                print('rs_std: %s' %rs_max_std)
                print('')
            elif setup['network']['language']['mode'] == 'discrete':
                train_loss_min = []
                val_loss_min = []
                prc_max = []
                auc_max = []
                for fold in range(k_count):
                    fold_train_loss_min = min(foldperf['fold{}'.format(fold+1)]['train_loss'])
                    fold_val_loss_min = min(foldperf['fold{}'.format(fold+1)]['val_loss'])
                    fold_prc_max = max(foldperf['fold{}'.format(fold+1)]['prc'])
                    fold_auc_max = max(foldperf['fold{}'.format(fold+1)]['roc_auc'])
                    train_loss_min.append(fold_train_loss_min)
                    val_loss_min.append(fold_val_loss_min)
                    prc_max.append(fold_prc_max)
                    auc_max.append(fold_auc_max)
                
                train_loss_min_std = np.std(train_loss_min)
                train_loss_min = np.mean(train_loss_min)
                val_loss_min_std = np.std(val_loss_min)
                val_loss_min = np.mean(val_loss_min)
                prc_max_std = np.std(prc_max)
                prc_max = np.mean(prc_max)
                auc_max_std = np.std(auc_max)
                auc_max = np.mean(auc_max)
                
                print('')
                print('train_loss_avg: %s' %train_loss_min)
                print('train_loss_std: %s' %train_loss_min_std)
                print('val_loss_avg: %s' %val_loss_min)
                print('val_loss_min_std: %s' %val_loss_min_std)
                print('prc_avg: %s' %prc_max)
                print('prc_std: %s' %prc_max_std)
                print('roc_auc_avg: %s' %auc_max)
                print('roc_auc_std: %s' %auc_max_std)
                print('')
        
        else:
            
            num_epochs = setup["training"]["num_epochs"]
            best_val_score = setup["training"]["best_val_score"]
            min_loss = setup["training"]["min_loss"]
            patience_train = 0
            patience_val = 0


            for epoch in tqdm(range(1, num_epochs + 1)): 
                train_score = train_model(fusion_model, optimizer, lr_function, dataloaders, setup)
                if setup['network']['language']['mode'] == 'continuous':
                    val_score, r2, pearson, spearman = validate_model(fusion_model, optimizer, dataloaders, setup)
                    writer.add_scalar('train/loss', train_score, epoch)
                    writer.add_scalar('train/patience', patience_train, epoch)
                    writer.add_scalar('val/loss', val_score, epoch)
                    writer.add_scalar('val/patience', patience_val, epoch)
                    writer.add_scalar('val/r2', r2, epoch)
                    writer.add_scalar('val/rp', pearson, epoch)
                    writer.add_scalar('val/rs', spearman, epoch)
                elif setup['network']['language']['mode'] == 'discrete':
                    val_score, acc, prc, auc = validate_model(fusion_model, optimizer, dataloaders, setup)
                    writer.add_scalar('train/loss', train_score, epoch)
                    writer.add_scalar('train/patience', patience_train, epoch)
                    writer.add_scalar('val/loss', val_score, epoch)
                    writer.add_scalar('val/patience', patience_val, epoch)
                    writer.add_scalar('val/prc', prc, epoch)
                    writer.add_scalar('val/roc_auc', auc, epoch)

                if setup["network"]["language"]["graph"]:
                    writer.add_scalar('val/lr_coder', optimizer.state_dict()['param_groups'][0]['lr'], epoch)
                    writer.add_scalar('val/lr_lang', optimizer.state_dict()['param_groups'][1]['lr'], epoch)
                else:
                    if setup["network"]["language"]["model"] == 'chemberta-77m-mlm' or setup["network"]["language"]["model"] == 'chemberta-10m-mlm' or setup["network"]["language"]["model"] == 'chemberta-77m-mtr' or setup["network"]["language"]["model"] == 'chemberta-10m-mtr':
                        writer.add_scalar('val/lr_ens_coder', optimizer.state_dict()['param_groups'][0]['lr'], epoch)
                        writer.add_scalar('val/lr_ens_transformer', optimizer.state_dict()['param_groups'][1]['lr'], epoch)
                        writer.add_scalar('val/lr_lang', optimizer.state_dict()['param_groups'][2]['lr'], epoch)
                    elif setup["network"]["language"]["model"] == 'chemberta-77m-mlm-only' or setup["network"]["language"]["model"] == 'chemberta-10m-mlm-only' or setup["network"]["language"]["model"] == 'chemberta-77m-mtr-only' or setup["network"]["language"]["model"] == 'chemberta-10m-mtr-only':
                        writer.add_scalar('val/lr_lang', optimizer.state_dict()['param_groups'][0]['lr'], epoch)
                    elif setup["network"]["language"]["model"] == "none":
                        writer.add_scalar('val/lr_ens_coder', optimizer.state_dict()['param_groups'][0]['lr'], epoch)
                        writer.add_scalar('val/lr_ens_transformer', optimizer.state_dict()['param_groups'][1]['lr'], epoch)

                if epoch % 100 == 0:
                    torch.save({'model_state_dict': fusion_model.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict()}, 
                    os.path.join(checkpoint_dir_run, 'SLEF_%s.pth' % (epoch)))
                
                if train_score <= min_loss:
                    min_loss = train_score
                    patience_train = 0
                    torch.save({'model_state_dict': fusion_model.state_dict(), 
                            'optimizer_state_dict': optimizer.state_dict(), 
                            'train_loss': train_score}, 
                            os.path.join(checkpoint_dir_run, 'SLEF.pth'))
                else:
                    patience_train += 1

                if best_val_score > val_score:
                    torch.save({'model_state_dict': fusion_model.state_dict(), 
                                'optimizer_state_dict': optimizer.state_dict(), 
                                'validation_loss': val_score}, 
                    os.path.join(checkpoint_dir_run, 'SLEF_validation.pth'))
                    best_val_score = val_score
                    patience_val = 0
                else:
                    patience_val += 1
                
                print('')
                print('epoch: %s' %epoch)
                print('min_loss: %s' %min_loss)
                print('train_patience: %s' %patience_train)
                print('best_val_score: %s' %best_val_score)
                print('validation_patience: %s' %patience_val)
                print('')

        writer.close()

    else:

        if setup["network"]["language"]["graph"] == False:
            config = {
                "network": {
                    "encoder": {
                        "in_channels": 1,
                        "num_layers" : tune.choice([1,2]),
                        "kernel_size": 3,
                        "n_hidden_fusion": tune.choice([16, 32, 64])
                    },
                    "transformer": {
                        "depth": tune.choice([1,2]),
                        "heads": tune.choice([1, 2, 4]),
                        "mlp_dim": tune.choice([16, 32, 64, 128]),
                        "dropout": tune.uniform(0.1, 0.5)
                    },
                    "decoder": {
                        "kernel_size": 1
                    },
                    "language": {
                        "model": setup["network"]["language"]["model"],
                        "graph":  setup["network"]["language"]["graph"],
                        "drop_rate": tune.uniform(0.1, 0.5),
                        "freeze": False,
                        "freeze_layer_count": tune.randint(1,3),
                        "mode": setup["network"]["language"]["mode"],
                        "num_predictions": setup["network"]["language"]["num_predictions"]
                    }
                },
                "learning_rates": {
                    "lr_ens_coder": tune.loguniform(1e-5, 1e-2),
                    "lr_ens_transformer": tune.loguniform(1e-6, 1e-3),
                    "lr_lang": tune.loguniform(1e-9, 1e-6),
                    "lr_strategy": 3,
                    #"lr_decay": tune.uniform(0.8, 0.95),
                    "lr_step":  tune.randint(1,5)
                }
            }
        elif setup["network"]["language"]["graph"]:
            config = {
                "network": {
                    "encoder": {
                        "in_channels": 1,
                        "num_layers" : 1, #tune.choice([1,2]),
                        "kernel_size": 3,
                        "n_hidden_fusion": 16, #tune.choice([16, 32, 64])
                    },
                    "transformer": {
                        "depth": 1, #tune.choice([1,2]),
                        "heads": 4, #tune.choice([1, 2, 4]),
                        "mlp_dim": 64, #tune.choice([16, 32, 64, 128]),
                        "dropout": 0.1 #tune.uniform(0.1, 0.5)
                    },
                    "decoder": {
                        "kernel_size": 1
                    },
                    "language": {
                        "model": setup["network"]["language"]["model"],
                        "graph": setup["network"]["language"]["graph"],
                        "drop_rate": tune.uniform(0.0, 0.1),
                        "freeze": False,
                        "freeze_layer_count": tune.randint(1,3),
                        "mode": setup["network"]["language"]["mode"],
                        "num_predictions": setup["network"]["language"]["num_predictions"]
                    }
                },
                "learning_rates": {
                    "lr_ens_coder": tune.loguniform(1e-5, 1e-2),
                    "lr_ens_transformer": 1e-6, #tune.loguniform(1e-6, 1e-3),
                    "lr_lang": tune.loguniform(1e-9, 1e-6),
                    "lr_strategy": 3,
                    #"lr_decay": tune.uniform(0.8, 0.95),
                    "lr_step":  tune.randint(1,5)
                }
            }
        
        # reporter to show on command line/output window
        if config["network"]["language"]["mode"] == 'continuous':
            reporter = CLIReporter(metric_columns=['training_iteration','validation_loss', 'r2','rp','rs'], max_report_frequency = 60)
        elif config["network"]["language"]["mode"] == 'discrete':
            reporter = CLIReporter(metric_columns=['training_iteration','validation_loss', 'prc','roc_auc'], max_report_frequency = 60)

        scheduler = HyperBandForBOHB(time_attr='training_iteration',
                                     max_t=setup["training"]["num_epochs"], # terminates hp trials at max epoch
                                     reduction_factor=4)
        
        asha_scheduler = ASHAScheduler(time_attr='training_iteration',
                                        max_t=setup["training"]["num_epochs"],
                                        grace_period=10,
                                        reduction_factor=2,
                                        brackets=1)
        
        # opt alg: https://docs.ray.io/en/latest/tune/api_docs/suggestion.html
        bayesopt = TuneBOHB(bohb_config = {"min_points_in_model" : None,
                                 "top_n_percent" : 15, "num_samples" : 64, "random_fraction" : 1/3,
                                 "bandwidth_factor" :3, "min_bandwidth" : 1e-3}, max_concurrent=1)
    
        #bayesopt = tune.search.ConcurrencyLimiter(bayesopt, max_concurrent=1)
        

        ray.init(_temp_dir="/path/to/temp/dir")
        
        
        tuner = tune.Tuner(tune.with_resources(tune.with_parameters(hp_tune, setup=setup), 
                resources={"cpu": len(os.sched_getaffinity(0)), "gpu": torch.cuda.device_count()}
                ),
                tune_config=tune.TuneConfig(
                    reuse_actors=False,
                    metric="validation_loss",
                    mode="min",
                    search_alg= bayesopt,
                    scheduler=scheduler,
                    num_samples= 50, # number of hyperparameter sets to try
                ),
                run_config=air.RunConfig(
                    name="ray_tune",
                    local_dir = logging_dir,
                    stop={"training_iteration": setup["training"]["num_epochs"]},
                    failure_config=air.FailureConfig(max_failures=3),
                    progress_reporter = reporter,
                    checkpoint_config=air.CheckpointConfig(num_to_keep=1, checkpoint_score_attribute="validation_loss", checkpoint_score_order="min")
                ),
                param_space=config,
            ).fit()
    
        
        print("best config:", str(tuner.get_best_result().config))
        