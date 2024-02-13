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

import torch
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from scipy import stats
from glob import glob
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import json
import os
from os.path import join
import numpy as np
import pandas as pd
from DeepNetworks.SLEF import SLEFNet
from torch.utils.data import DataLoader
from ray.air.checkpoint import Checkpoint
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
import pdb
from DataLoader import SuperSet, SuperSet_csv, get_patch
from tqdm import tqdm
from utils import getDataSetDirectories, collateFunction, get_loss
import itertools
import ablang
import sys

def patch_iterator(img, positions, size):
    """Iterator across square patches of `ens map` located in `positions`."""
    for x, y in positions:
        yield get_patch(img=img, x=x, y=y, size=size)

def data_import(dataset):

    data_directory = setup["paths"]["data_directory"]
    print(data_directory)
    if not setup["network"]["language"]["graph"]:
        holdout_list = getDataSetDirectories(setup, os.path.join(data_directory, "holdout"))
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

        holdout_dataset = SuperSet(superset_dir=holdout_list,
                                        setup=setup["training"],
                                        ens_L=ens_L, lang=lang, graph=graph, chemprop=chemprop, median_ref=median_ref)

        holdout_dataloader = DataLoader(holdout_dataset,
                                    batch_size=int(1),
                                    shuffle=False,
                                    num_workers=n_workers,
                                    collate_fn=collateFunction(setup=setup, set_L=set_L),
                                    pin_memory=True)


        dataloaders = {'holdout': holdout_dataloader}
    else:
        if dataset:
            data = pd.read_csv(os.path.join(data_directory,dataset))
            holdout_dataset = SuperSet_csv(data=data, X = 'ids', y ='y')
        else:
            raise ValueError("dataset not found in specified directory: %s" % str(os.path.join(data_directory, dataset)))
    
        holdout_dataloader = DataLoader(holdout_dataset,
                                    batch_size=int(1),
                                    shuffle=False,
                                    num_workers=n_workers,
                                    collate_fn=collateFunction(setup=setup, set_L=set_L),
                                    pin_memory=True)
        dataloaders = {'holdout': holdout_dataloader}


    torch.cuda.empty_cache()
    return dataloaders

def test_model(fusion_model, filename, dataloaders, setup):
    
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    fusion_model.eval()

    PROPs = []
    PREDs = []
    holdout_score = 0

    for lrs, ens_attention_masks, alphas, tokens, attention_masks, props, names in tqdm(dataloaders['holdout']):
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
        # compute loss for validation set
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
        batch_score = score.detach().cpu().numpy() * len(props) / len(dataloaders['holdout'].dataset)
        
        #if setup['training']["kfold"]:
        #    batch_score = score.detach().cpu().numpy() * len(props) / len(dataloaders['holdout'].sampler.indices)
        #else:
         #   batch_score = score.detach().cpu().numpy() * len(props) / len(dataloaders['holdout'].dataset)
            
            #if setup['training']['combine_losses']:
            #    score1 = get_loss(prop_preds[i], props[i], metric='L1')
            #    score2 = get_loss(prop_preds[i], props[i], metric='L2')
             #   score += setup['training']['alpha1'] * score1 + setup['training']['alpha2'] * score2
            #else:
             #   score += get_loss(prop_preds[i],props[i], metric=setup['training']['loss'], num_predictions=setup['network']['language']['num_predictions'])

        #batch_score = score.detach().cpu().numpy() * len(props) / len(dataloaders['holdout'].dataset)
        holdout_score += batch_score

    PROPs = np.squeeze(np.array(PROPs))
    PREDs = np.squeeze(np.array(PREDs))

    if setup['network']['language']['mode'] == 'continuous':
        r2 = r2_score(PROPs, PREDs)
        pearson = stats.pearsonr(PROPs, PREDs).statistic
        spearman = stats.spearmanr(PROPs, PREDs).correlation

        if setup["training"]["kfold"]:
            return holdout_score, r2, pearson, spearman, PROPs, PREDs
        else:
            # plot predictions vs properties
            fig, ax = plt.subplots()
            ax.scatter(PROPs, PREDs, c='black',edgecolors=(0, 0, 0))
            ax.plot([PROPs.min(), PROPs.max()], [PROPs.min(), PROPs.max()], 'k--', lw=4)
            ax.set_xlabel('Measured')
            ax.set_ylabel('Predicted')
            ax.set_title('R\u00b2: '+ "{:.2f}".format(r2))
            text_box2 = AnchoredText('R\u00b2: ' "{:.2f}".format(r2) + "\n" + '$r_{p}$: ' + "{:.2f}".format(pearson) + "\n" + '$r_{s}$: ' + "{:.2f}".format(spearman), frameon=True,prop=dict(color = 'black'), loc=4, pad=0.5)
            plt.setp(text_box2.patch, facecolor='white', alpha=0.5)
            plt.gca().add_artist(text_box2)
            plt.savefig(r'%sholdout_%s.png' % (path + "/", filename), dpi=300, bbox_inches='tight')
            return holdout_score, r2, pearson, spearman, PROPs, PREDs
        
    elif setup['network']['language']['mode'] == 'discrete':
        if  setup['network']['language']['num_predictions'] == 1:
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
            print('val_score:  ', holdout_score)
            print('')
        elif setup['network']['language']['num_predictions'] > 2:
            PREDs = PREDs.argmax(axis=1)
            acc = None
            ap = average_precision_score(PROPs, PREDs)
            auc = roc_auc_score(PROPs, PREDs)
            print('')
            print('prc: ', ap, 'roc_auc: ', auc)
            print('val_score:  ', holdout_score)
            print('')
        return holdout_score, acc, ap, auc, PROPs, PREDs
    

    

def main(dataset, path, model_path, filename, epoch_num, validation, ray_tune, checkpoint, setup):
    
    # random seeds for reproducibility
    np.random.seed(setup["training"]["seed"]) 
    torch.manual_seed(setup["training"]["seed"])

    # define compute specifications
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # import holdout data
    dataloaders = data_import(dataset)

    if ray_tune:
        if setup["training"]["kfold"]:
            with open(model_path + 'ray_tune/' + ray_tune + '/params.json', "r") as read_file:
                config = json.load(read_file)
            checkpoint = Checkpoint.from_directory(model_path + 'ray_tune/' + ray_tune + '/' + checkpoint + '/')
            checkpoint_dict = checkpoint.to_dict()
            checkpoint_dict = checkpoint_dict.get("model_state_dict")
            for fold in tqdm(range(len(checkpoint_dict))):
                checkpoint_dict[fold] = {k[7:]: v for k, v in checkpoint_dict[fold].items()}
            if setup['network']['language']['mode'] == 'continuous':
                cv_data = {'holdout_score': [],'r2':[],'rp':[],'rs':[], 'props':[], 'preds':[]}
            elif setup['network']['language']['mode'] == 'discrete':
                cv_data = {'holdout_score': [], 'prc':[],'roc_auc':[], 'props':[], 'preds':[]}
            for fold in tqdm(range(setup["training"]["kfold"])):
                fusion_model = SLEFNet(config["network"])
                #checkpoint_dict[fold] = {k[7:]: v for k, v in checkpoint_dict[fold].items()}
                fusion_model.load_state_dict(checkpoint_dict[fold])
                fusion_model.eval()
                fusion_model = torch.nn.DataParallel(fusion_model, device_ids=[0]).to(device)        
                #fusion_model = torch.nn.DataParallel(fusion_model, device_ids=list(range(torch.cuda.device_count()))).to(device)
                #fusion_model.to(device)
                # append props and preds to cv_data
                if setup['network']['language']['mode'] == 'continuous':
                    holdout_score, r2, pearson, spearman, props, preds = test_model(fusion_model, filename, dataloaders, setup)
                    cv_data['props'].append(props)
                    cv_data['preds'].append(preds)
                    cv_data['holdout_score'].append(holdout_score)
                    cv_data['r2'].append(r2)
                    cv_data['rp'].append(pearson)
                    cv_data['rs'].append(spearman)
                    print('r2: ', r2, 'r_p: ', pearson, 'r_s: ', spearman)
                    print('')
                    print('holdout_score: ', holdout_score)
                elif setup['network']['language']['mode'] == 'discrete':
                    holdout_score, acc, ap, auc, props, preds = test_model(fusion_model, filename, dataloaders, setup)
                    cv_data['props'].append(props)
                    cv_data['preds'].append(preds)
                    cv_data['holdout_score'].append(holdout_score)
                    cv_data['prc'].append(ap)
                    cv_data['roc_auc'].append(auc)
                    print('prc: ', ap, 'roc_auc: ', auc)
                    print('')
                    print('holdout_score: ', holdout_score)

            if setup['network']['language']['mode'] == 'continuous': 
                print('holdout_score_avg: ', np.mean(cv_data['holdout_score']))
                print('holdout_score_std: ', np.std(cv_data['holdout_score']))
                print('r2_avg: ', np.mean(cv_data['r2']))
                print('r2_std: ', np.std(cv_data['r2']))
                print('rp_avg: ', np.mean(cv_data['rp']))
                print('rp_std: ', np.std(cv_data['rp']))
                print('rs_avg: ', np.mean(cv_data['rs']))
                print('rs_std: ', np.std(cv_data['rs']))
                # average across props and preds sublists
                cv_data['props_avg'] = np.mean(cv_data['props'], axis=0)
                cv_data['preds_avg'] = np.mean(cv_data['preds'], axis=0)
                # std across props and preds sublists
                cv_data['props_std'] = np.std(cv_data['props'], axis=0)
                cv_data['preds_std'] = np.std(cv_data['preds'], axis=0)
                # plot predictions vs properties as scatter plot with error bars
                fig, ax = plt.subplots()
                ax.errorbar(cv_data['props_avg'], cv_data['preds_avg'], xerr=cv_data['props_std'], yerr=cv_data['preds_std']/np.sqrt(setup["training"]["kfold"]), fmt='o', color='black', ecolor='black', elinewidth=3, capsize=0)
                ax.set_xlabel('Measured')
                ax.set_ylabel('Predicted')
                ax.plot([cv_data['props_avg'].min(), cv_data['props_avg'].max()], [cv_data['props_avg'].min(), cv_data['props_avg'].max()], 'k--', lw=4)
                text_box2 = AnchoredText('R\u00b2: ' "{:.2f}".format(np.mean(cv_data['r2'])) + " " +  u"\u00B1" + " {:.2f}".format(np.std(cv_data['r2'])/np.sqrt(setup["training"]["kfold"])) +
                                        "\n" + '$r_{p}$: ' + "{:.2f}".format(np.mean(cv_data['rp'])) + " " + u"\u00B1" + " {:.2f}".format(np.std(cv_data['rp'])/np.sqrt(setup["training"]["kfold"])) +
                                        "\n" + '$r_{s}$: ' + "{:.2f}".format(np.mean(cv_data['rs'])) + " " + u"\u00B1" + " {:.2f}".format(np.std(cv_data['rs'])/np.sqrt(setup["training"]["kfold"])),
                                        frameon=True,prop=dict(color = 'black'), loc=4, pad=0.5)
                plt.setp(text_box2.patch, facecolor='white', alpha=0.5)
                plt.gca().add_artist(text_box2)
                plt.savefig(r'%sholdout_%s.png' % (path + "/", filename), dpi=300, bbox_inches='tight')
            elif setup['network']['language']['mode'] == 'discrete':
                print('holdout_score_avg: ', np.mean(cv_data['holdout_score']))
                print('holdout_score_std: ', np.std(cv_data['holdout_score']))
                print('prc_avg: ', np.mean(cv_data['prc']))
                print('prc_std: ', np.std(cv_data['prc']))
                print('roc_auc_avg: ', np.mean(cv_data['roc_auc']))
                print('roc_auc_std: ', np.std(cv_data['roc_auc']))

        else:
            with open(model_path + 'ray_tune/' + ray_tune + '/params.json', "r") as read_file:
                config = json.load(read_file)
            fusion_model = SLEFNet(config["network"])
            checkpoint = Checkpoint.from_directory(model_path + 'ray_tune/' + ray_tune + '/' + checkpoint + '/')
            checkpoint_dict = checkpoint.to_dict()
            checkpoint_dict = checkpoint_dict.get("model_state_dict")
            consume_prefix_in_state_dict_if_present(checkpoint_dict, "module.")
            fusion_model.load_state_dict(checkpoint_dict)
            fusion_model = torch.nn.DataParallel(fusion_model, device_ids=[0]).to(device)      
            #fusion_model = torch.nn.DataParallel(fusion_model, device_ids=list(range(torch.cuda.device_count()))).to(device)
            #fusion_model.to(device)
            holdout_score, r2, pearson, spearman, props, preds = test_model(fusion_model, filename, dataloaders, setup)
            print('r2: ', r2, 'r_p: ', pearson, 'r_s: ', spearman)
            print('')
            print('holdout_score: ', holdout_score)

    else:
        if epoch_num:
            fusion_model = SLEFNet(setup["network"])
            module_state_dict = torch.load(os.path.join(model_path,
                                                    'SLEF_'+str(epoch_num)+'.pth'))
            module_state_dict = module_state_dict["model_state_dict"]
            model_state_dict = {k[7:]: v for k, v in module_state_dict.items()}
            fusion_model.load_state_dict(model_state_dict)
            fusion_model = torch.nn.DataParallel(fusion_model, device_ids=[0]).to(device)        
            #fusion_model = torch.nn.DataParallel(fusion_model, device_ids=list(range(torch.cuda.device_count()))).to(device)
            #fusion_model.to(device)
            if setup['network']['language']['mode'] == 'continuous':
                holdout_score, r2, pearson, spearman, props, preds = test_model(fusion_model, filename, dataloaders, setup)
                print('r2: ', r2, 'r_p: ', pearson, 'r_s: ', spearman)
                print('')
                print('holdout_score: ', holdout_score)
            elif setup['network']['language']['mode'] == 'discrete':
                holdout_score, acc, ap, auc, props, preds = test_model(fusion_model, filename, dataloaders, setup)
                print('prc: ', ap, 'roc_auc: ', auc)
                print('holdout_score: ', holdout_score)


        elif validation:
            print("validation")
            fusion_model = SLEFNet(setup["network"])
            module_state_dict = torch.load(os.path.join(model_path,'SLEF_validation.pth'))
            module_state_dict = module_state_dict["model_state_dict"]
            model_state_dict = {k[7:]: v for k, v in module_state_dict.items()}
            fusion_model.load_state_dict(model_state_dict)
            fusion_model = torch.nn.DataParallel(fusion_model, device_ids=[0]).to(device)        
            #fusion_model = torch.nn.DataParallel(fusion_model, device_ids=list(range(torch.cuda.device_count()))).to(device)
            #fusion_model.to(device)
            if setup['network']['language']['num_predictions'] == 1:
                holdout_score, r2, pearson, spearman, props, preds = test_model(fusion_model, filename, dataloaders, setup)
                print('r2: ', r2, 'r_p: ', pearson, 'r_s: ', spearman)
                print('')
                print('holdout_score: ', holdout_score)
            elif setup['network']['language']['num_predictions'] > 1:
                holdout_score, acc, ap, auc, props, preds = test_model(fusion_model, filename, dataloaders, setup)
                print('prc: ', ap, 'roc_auc: ', auc)
                print('holdout_score: ', holdout_score)
        elif setup["training"]["kfold"]:
            if setup['network']['language']['mode'] == 'continuous':
                cv_data = {'holdout_score': [],'r2':[],'rp':[],'rs':[],'props':[], 'preds':[]}
            elif setup['network']['language']['mode'] == 'discrete':
                cv_data = {'holdout_score': [],'prc':[],'roc_auc':[],'props':[], 'preds':[]}
            for fold in tqdm(range(setup["training"]["kfold"])):
                fusion_model = SLEFNet(setup["network"])
                module_state_dict = torch.load(os.path.join(model_path,'SLEF_f%s_validation.pth')%str(fold+1))
                module_state_dict = module_state_dict["model_state_dict"]
                model_state_dict = {k[7:]: v for k, v in module_state_dict.items()}
                fusion_model.load_state_dict(model_state_dict)
                fusion_model = torch.nn.DataParallel(fusion_model, device_ids=[0]).to(device)        
                #fusion_model = torch.nn.DataParallel(fusion_model, device_ids=list(range(torch.cuda.device_count()))).to(device)
                #fusion_model.to(device)
                if setup['network']['language']['mode'] == 'continuous':
                    holdout_score, r2, pearson, spearman, props, preds = test_model(fusion_model, filename, dataloaders, setup)
                    # props and preds are numpy arrays to list
                    # append props and preds to cv_data
                    cv_data['props'].append(props)
                    cv_data['preds'].append(preds)
                    cv_data['holdout_score'].append(holdout_score)
                    cv_data['r2'].append(r2)
                    cv_data['rp'].append(pearson)
                    cv_data['rs'].append(spearman)
                    print('r2: ', r2, 'r_p: ', pearson, 'r_s: ', spearman)
                    print('')
                    print('holdout_score: ', holdout_score)
                elif setup['network']['language']['mode'] == 'discrete':
                    holdout_score, acc, ap, auc, props, preds = test_model(fusion_model, filename, dataloaders, setup)
                    # props and preds are numpy arrays to list
                    # append props and preds to cv_data
                    cv_data['props'].append(props)
                    cv_data['preds'].append(preds)
                    cv_data['holdout_score'].append(holdout_score)
                    cv_data['prc'].append(ap)
                    cv_data['roc_auc'].append(auc)
                    print('prc: ', ap, 'roc_auc: ', auc)
                    print('holdout_score: ', holdout_score)
            
            if setup['network']['language']['mode'] == 'continuous':
                print('holdout_score_avg: ', np.mean(cv_data['holdout_score']))
                print('holdout_score_std: ', np.std(cv_data['holdout_score']))
                print('r2_avg: ', np.mean(cv_data['r2']))
                print('r2_std: ', np.std(cv_data['r2']))
                print('rp_avg: ', np.mean(cv_data['rp']))
                print('rp_std: ', np.std(cv_data['rp']))
                print('rs_avg: ', np.mean(cv_data['rs']))
                print('rs_std: ', np.std(cv_data['rs']))
                # average across props and preds sublists
                cv_data['props_avg'] = np.mean(cv_data['props'], axis=0)
                cv_data['preds_avg'] = np.mean(cv_data['preds'], axis=0)
                # std across props and preds sublists
                cv_data['props_std'] = np.std(cv_data['props'], axis=0)
                cv_data['preds_std'] = np.std(cv_data['preds'], axis=0)
                # plot predictions vs properties as scatter plot with error bars
                fig, ax = plt.subplots()
                ax.errorbar(cv_data['props_avg'], cv_data['preds_avg'], xerr=cv_data['props_std'], yerr=cv_data['preds_std']/np.sqrt(setup["training"]["kfold"]), fmt='o', color='black', ecolor='black', elinewidth=3, capsize=0)
                ax.set_xlabel('Measured')
                ax.set_ylabel('Predicted')
                ax.plot([cv_data['props_avg'].min(), cv_data['props_avg'].max()], [cv_data['props_avg'].min(), cv_data['props_avg'].max()], 'k--', lw=4)
                text_box2 = AnchoredText('R\u00b2: ' "{:.2f}".format(np.mean(cv_data['r2'])) + " " +  u"\u00B1" + " {:.2f}".format(np.std(cv_data['r2'])/np.sqrt(setup["training"]["kfold"])) +
                                        "\n" + '$R_{p}$: ' + "{:.2f}".format(np.mean(cv_data['rp'])) + " " + u"\u00B1" + " {:.2f}".format(np.std(cv_data['rp'])/np.sqrt(setup["training"]["kfold"])) +
                                        "\n" + '$R_{s}$: ' + "{:.2f}".format(np.mean(cv_data['rs'])) + " " + u"\u00B1" + " {:.2f}".format(np.std(cv_data['rs'])/np.sqrt(setup["training"]["kfold"])),
                                        frameon=True,prop=dict(color = 'black'), loc=4, pad=0.5)
                plt.setp(text_box2.patch, facecolor='white', alpha=0.5)
                plt.gca().add_artist(text_box2)
                plt.savefig(r'%sholdout_%s.png' % (path + "/", filename), dpi=300, bbox_inches='tight')
            elif setup['network']['language']['mode'] == 'discrete':
                print('holdout_score_avg: ', np.mean(cv_data['holdout_score']))
                print('holdout_score_std: ', np.std(cv_data['holdout_score']))
                print('prc_avg: ', np.mean(cv_data['prc']))
                print('prc_std: ', np.std(cv_data['prc']))
                print('roc_auc_avg: ', np.mean(cv_data['roc_auc']))
                print('roc_auc_std: ', np.std(cv_data['roc_auc']))

        else:
            fusion_model = SLEFNet(setup["network"])
            module_state_dict = torch.load(os.path.join(model_path,'SLEF.pth'))
            module_state_dict = module_state_dict["model_state_dict"]
            model_state_dict = {k[7:]: v for k, v in module_state_dict.items()}
            fusion_model.load_state_dict(model_state_dict)
            fusion_model = torch.nn.DataParallel(fusion_model, device_ids=[0]).to(device)        
            #fusion_model = torch.nn.DataParallel(fusion_model, device_ids=list(range(torch.cuda.device_count()))).to(device)
            #fusion_model.to(device)
            if setup['network']['language']['mode'] == 'continuous':
                holdout_score, r2, pearson, spearman, props, preds = test_model(fusion_model, filename, dataloaders, setup)
                print('r2: ', r2, 'r_p: ', pearson, 'r_s: ', spearman)
                print('')
                print('holdout_score: ', holdout_score)
            elif setup['network']['language']['mode'] == 'discrete':
                holdout_score, acc, ap, auc, props, preds = test_model(fusion_model, filename, dataloaders, setup)
                print('prc: ', ap, 'roc_auc: ', auc)
                print('holdout_score: ', holdout_score)
            

if __name__ == '__main__':

    with open('./config/setup.json', "r") as read_file:
        setup = json.load(read_file)
    
    dataset = setup["holdout"]["dataset"]
    path = setup["holdout"]["holdout_data_path"]
    model_path = setup["holdout"]["model_path"]
    filename = setup["holdout"]["filename"]
    epoch_num = setup["holdout"]["epoch_num"]
    ray_tune = setup["holdout"]["ray_tune"]
    checkpoint = setup["holdout"]["checkpoint"]
    validation = setup["holdout"]["validation"]

    with open(model_path + 'setup.json', "r") as read_file:
        setup = json.load(read_file)

    main(dataset, path, model_path, filename, epoch_num, validation, ray_tune, checkpoint, setup)