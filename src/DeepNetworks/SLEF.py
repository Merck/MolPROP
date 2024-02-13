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

""" MolPROP a neural netowrk for fusion of small molecule graph & language. """

import torch
import torch.nn.functional as F
import sys
from einops import rearrange, repeat
from torch import nn
import pdb
from torch_scatter import scatter_mean
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import GATv2Conv
from transformers.models.roberta import RobertaModel, RobertaTokenizer
import os
from typing import Tuple

def _chemberta_forward(
    chemberta_model=None,
    tokens=None,
    attention_masks=None
):
    """forward pass throught pretrained chemberta to get token-level embeddings
    Arguments:
        chemberta_model: pretrained SMILES string model
        embeding_dim: dimension of the embeddings (chemberta = 384)
        tokens: tokenizations of the input strings
    Returns:
        output token-level atom embeddings (batch_len, token_len, embedding_dim)
    Sources:
        language model backpropogation: https://github.com/aws-samples/lm-gvp
        chemberta models: https://huggingface.co/DeepChem
    """
    token_embeddings = chemberta_model(tokens, attention_masks).last_hidden_state[:, 1:-1, :]

    return token_embeddings

def _freeze_chemberta(
    chemberta_model=None, freeze_chemberta=True, freeze_layer_count=-1
):
    """freeze pretrained parameters in ChemBERTa model
    Arguments:
        chemberta_model: pretrained SMILES string model
        freeze: Bool to freeze the pretrained chemberta model
        freeze_layer_count: If freeze == False, number of layers to freeze (max_layers = 3).
    Returns:
        chemberta model w/ annotated differentiable parameters
    Sources:
        language model backpropogation: https://github.com/aregre-samples/lm-gvp
        chemberta models: https://huggingface.co/DeepChem
    """
    if freeze_chemberta:
        # freeze the entire chemberta_model model
        for param in chemberta_model.parameters():
            param.requires_grad = False
    else:
        if freeze_layer_count != -1:
            # freeze layers in encoder
            for layer in chemberta_model.encoder.layer[:freeze_layer_count]: # 0-3 layers in ChemBERTa-2 encoder
                for param in layer.parameters():
                    param.requires_grad = False
    return None


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5 # 1/sqrt(64)=0.125
        self.to_qkv = nn.Linear(dim, dim*3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None, maps = None, fusion_size=None):
        #print(maps)
        b, n, _, h = *x.shape, self.heads # b:batch_size, n:17, _:64, heads:heads as an example
        #print('attention, tensor(B, L, C)')
        #print(x.shape)
        qkv = self.to_qkv(x).chunk(3, dim = -1) # self.to_qkv(x) to generate [b=batch_size, n=17, hd=192]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv) # q, k, v [b=batch_size, heads=heads, n=17, d=depth]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale # [b=batch_size, heads=heads, 17, 17]

        mask_value = -torch.finfo(dots.dtype).max # A big negative number

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True) # [b=batch_size, 17]
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions' # mask [4, 17], dots [4, 8, 17, 17]
            assert len(mask.shape) == 2
            dots = dots.view(-1, fusion_size*fusion_size, dots.shape[1], dots.shape[2], dots.shape[3])
            mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
           # print('mask:') 
           # print(mask.shape)
           # print('dots:')
           # print(dots.shape)
            dots = dots * mask + mask_value * (1 - mask)
            dots = dots.view(-1, dots.shape[2], dots.shape[3], dots.shape[4])
            del mask

        if maps is not None:
             #maps [16384, 16] -> [16384, 17] , dots [16384, 8, 17, 17]
            maps = F.pad(maps.flatten(1), (1, 0), value = 1.)
            maps = maps.unsqueeze(1).unsqueeze(2)
            dots.masked_fill_(~maps.bool(), mask_value)
            del maps

        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        #print(out.shape)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None, maps = None, fusion_size=None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask, maps = maps, fusion_size = fusion_size)
            x = ff(x)
        return x

class SuperResTransformer(nn.Module):
    def __init__(self, *, dim, depth, heads, mlp_dim, dropout = 0.1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.to_cls_token = nn.Identity()

    def forward(self, img, mask = None, maps= None, fusion_size = None):
        b, n, _ = img.shape
        # No need to add position code, just add token
        features_token = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((features_token, img), dim=1)
        x = self.transformer(x, mask, maps, fusion_size)
        x = self.to_cls_token(x[:, 0])

        return x

class ResidualBlock(nn.Module):
    def __init__(self, n_hidden_fusion=64, kernel_size=3):
        '''
        Args:
            n_hidden_fusion : int, number of hidden channels
            kernel_size : int, shape of a 2D kernel
        '''
        super(ResidualBlock, self).__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=n_hidden_fusion, out_channels=n_hidden_fusion, kernel_size=kernel_size, padding=padding),
            nn.PReLU(),
            nn.Conv2d(in_channels=n_hidden_fusion, out_channels=n_hidden_fusion, kernel_size=kernel_size, padding=padding),
            nn.PReLU()
        )

    def forward(self, x):
        '''
        Args:
            x : tensor (B, C, W, H), hidden state
        Returns:
            x + residual: tensor (B, C, W, H), new hidden state
        '''
        residual = self.block(x)
        return x + residual

class Encoder(nn.Module):
    def __init__(self, setup):
        '''
        Args:
            setup : dict, setupuration file
        '''
        super(Encoder, self).__init__()

        in_channels = setup["in_channels"]
        num_layers = setup["num_layers"]
        kernel_size = setup["kernel_size"]
        n_hidden_fusion = setup["n_hidden_fusion"]
        padding = kernel_size // 2

        self.init_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=n_hidden_fusion, kernel_size=kernel_size, padding=padding),
            nn.PReLU())

        res_layers = [ResidualBlock(n_hidden_fusion, kernel_size) for _ in range(num_layers)]
        self.res_layers = nn.Sequential(*res_layers)

        self.final = nn.Sequential(
            nn.Conv2d(in_channels=n_hidden_fusion, out_channels=n_hidden_fusion, kernel_size=kernel_size, padding=padding)
        )

    def forward(self, x):
        '''
        Encodes an input tensor x.
        Args:
            x : tensor (B, C_in, W, H), input images
        Returns:
            out: tensor (B, C, W, H), hidden states
        '''
        #print('encoder')
        #print(x.shape)
        x = self.init_layer(x)
        x = self.res_layers(x)
        x = self.final(x)
       # print(x.shape)
        return x

class Decoder(nn.Module):
    def __init__(self, setup):
        '''
        Args:
            setup : dict, setupuration file
        '''
        super(Decoder, self).__init__()

        self.final = nn.Sequential(nn.Conv2d(in_channels=setup["encoder"]["n_hidden_fusion"],
                                             out_channels=1,
                                             kernel_size=setup["decoder"]["kernel_size"],
                                             padding=setup["decoder"]["kernel_size"] // 2),
                     nn.PReLU())

        #self.pixelshuffle = nn.PixelShuffle(1)

    def forward(self, x):
 
        x = self.final(x)
        #print('decoder')
        #print(x.shape)
        #x = self.pixelshuffle(x)
       # print(x.shape)

        return x
    
class LangGCN(nn.Module):
    def __init__(self, setup):
        """Language + GCN structure fusion. (modified from LM-GVP)


        Args:
            setup : dict, setup file (i.e., setup/setup.json)

        Returns:
            None
        """
        super(LangGCN, self).__init__()

        self.drop_rate = setup['drop_rate']
        self.seq_len = 160
        self.n_hidden = 512
        self.model = setup['model']
        self.freeze = setup['freeze']
        self.freeze_layer_count = setup['freeze_layer_count']
        self.mode = setup['mode']
        self.num_predictions = setup['num_predictions']

        if setup['model'] == 'chemberta-77m-mlm':
            self.chemberta =  RobertaModel.from_pretrained('config/chembert/ChemBERTa-77M-MLM')
            self.embedding_dim = 384
            _freeze_chemberta(
            self.chemberta,
            freeze_chemberta = setup['freeze'],
            freeze_layer_count = setup['freeze_layer_count'])

        elif setup['model'] == 'chemberta-10m-mlm':
            self.chemberta =  RobertaModel.from_pretrained('config/chembert/ChemBERTa-10M-MLM')
            self.embedding_dim = 384
            _freeze_chemberta(
            self.chemberta,
            freeze_chemberta = setup['freeze'],
            freeze_layer_count = setup['freeze_layer_count'])

        
    
        elif setup['model'] == 'chemberta-77m-mtr':
            self.chemberta =  RobertaModel.from_pretrained('config/chembert/ChemBERTa-77M-MTR')
            self.embedding_dim = 384
            _freeze_chemberta(
            self.chemberta,
            freeze_chemberta = setup['freeze'],
            freeze_layer_count = setup['freeze_layer_count'])

        
        elif setup['model'] == 'chemberta-10m-mtr':
            self.chemberta =  RobertaModel.from_pretrained('config/chembert/ChemBERTa-10M-MTR')
            self.embedding_dim = 384
            _freeze_chemberta(
            self.chemberta,
            freeze_chemberta = setup['freeze'],
            freeze_layer_count = setup['freeze_layer_count'])
        
        elif setup['model'] == 'chemberta-77m-mlm-only' or setup['model'] == 'chemberta-10m-mlm-only' or setup['model'] == 'chemberta-77m-mtr-only' or setup['model'] == 'chemberta-10m-mtr-only' or setup['model'] == 'none':
            raise ValueError('set graph == False for language-only models')
        
        self.conv1 = GCNConv(self.embedding_dim, 128)
        self.conv2 = GCNConv(128, 384)
        self.conv3 = GCNConv(384, 512)
        self.conv4 = GCNConv(512, 1024)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=self.drop_rate)

        self.dense = nn.Sequential(
            nn.Linear(2048, self.n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_hidden, self.num_predictions),
        )

        if self.mode == 'discrete':
            if self.num_predictions == 1:
                self.final_layer = nn.Identity()
                #self.final_layer = nn.Sigmoid()
            elif self.num_predictions > 2:
                self.final_layer = nn.Softmax(dim=1)
            else:
                raise ValueError('invalid number of predictions')
        elif self.mode == 'continuous':
            self.final_layer = nn.Identity()


    def forward(self, lrs, tokens, attention_masks, fusion_size=None): 

        edge_index = lrs.edge_index
        # get number of nodes in lrs
        n_nodes = lrs.num_nodes
        batch_size = lrs.num_graphs
        tokens = tokens
        attention_masks = attention_masks

        if self.model == 'chemberta-77m-mlm' or self.model == 'chemberta-10m-mlm' or self.model == 'chemberta-77m-mtr' or self.model == 'chemberta-10m-mtr':
            token_embeddings = _chemberta_forward(
                self.chemberta,
                tokens, attention_masks)
            # extract heavy atom embeddings from SMILES token embeddings via custom attention mask
            attention_mask_1d = attention_masks[:,1:-1].reshape(-1)
            atom_embeddings = token_embeddings.reshape(-1, self.embedding_dim)[attention_mask_1d == 1]
        else:
            raise ValueError("invalid SMILES language model")
        
         # GCN forward
        conv1_out = self.conv1(atom_embeddings, edge_index)
        conv2_out = self.conv2(conv1_out, edge_index)
        conv3_out = self.conv3(conv2_out, edge_index)
        conv4_out = self.conv4(conv3_out, edge_index)
        # residual concat
        out = torch.cat((conv1_out, conv2_out, conv3_out, conv4_out), dim=-1)
        out = self.dropout(self.relu(out))  # [n_nodes, 2048]
        # aggregate node vectors to graph
        out = scatter_mean(out, lrs.batch, dim=0)  # [bs, 2048]
        p = self.dense(out).squeeze(-1) + 0.5
        p = self.final_layer(p)
        return p.unsqueeze(1)  # [bs, 1]

class LangGAT(nn.Module):
    def __init__(self, setup):
        """Language + GAT structure fusion. (modified from LM-GVP)


        Args:
            setup : dict, setup file (i.e., setup/setup.json)

        Returns:
            None
        """
        super(LangGAT, self).__init__()

        self.drop_rate = setup['drop_rate']
        self.seq_len = 160
        self.n_hidden = 512
        self.model = setup['model']
        self.freeze = setup['freeze']
        self.freeze_layer_count = setup['freeze_layer_count']
        self.mode = setup['mode']
        self.num_predictions = setup['num_predictions']

        if setup['model'] == 'chemberta-77m-mlm':
            self.chemberta =  RobertaModel.from_pretrained('config/chembert/ChemBERTa-77M-MLM')
            self.embedding_dim = 384
            _freeze_chemberta(
            self.chemberta,
            freeze_chemberta = setup['freeze'],
            freeze_layer_count = setup['freeze_layer_count'])

        elif setup['model'] == 'chemberta-10m-mlm':
            self.chemberta =  RobertaModel.from_pretrained('config/chembert/ChemBERTa-10M-MLM')
            self.embedding_dim = 384
            _freeze_chemberta(
            self.chemberta,
            freeze_chemberta = setup['freeze'],
            freeze_layer_count = setup['freeze_layer_count'])
    
        elif setup['model'] == 'chemberta-77m-mtr':
            self.chemberta =  RobertaModel.from_pretrained('config/chembert/ChemBERTa-77M-MTR')
            self.embedding_dim = 384
            _freeze_chemberta(
            self.chemberta,
            freeze_chemberta = setup['freeze'],
            freeze_layer_count = setup['freeze_layer_count'])
        
        elif setup['model'] == 'chemberta-10m-mtr':
            self.chemberta =  RobertaModel.from_pretrained('config/chembert/ChemBERTa-10M-MTR')
            self.embedding_dim = 384
            _freeze_chemberta(
            self.chemberta,
            freeze_chemberta = setup['freeze'],
            freeze_layer_count = setup['freeze_layer_count'])
        
        elif setup['model'] == 'chemberta-77m-mlm-only' or setup['model'] == 'chemberta-10m-mlm-only' or setup['model'] == 'chemberta-77m-mtr-only' or setup['model'] == 'chemberta-10m-mtr-only' or setup['model'] == 'none':
            raise ValueError('set graph == False for language-only models')

        self.conv1 = GATConv(self.embedding_dim, 128, 4)
        self.conv2 = GATConv(512, 128, 4)
        self.conv3 = GATConv(512, 256, 4)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=self.drop_rate)

        self.dense = nn.Sequential(
            nn.Linear(2048, self.n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_hidden, self.num_predictions),
        )

        if self.mode == 'discrete':
            if self.num_predictions == 1:
                self.final_layer = nn.Identity()
                #self.final_layer = nn.Sigmoid()
            elif self.num_predictions > 2:
                self.final_layer = nn.Softmax(dim=1)
            else:
                raise ValueError('invalid number of predictions')
        elif self.mode == 'continuous':
            self.final_layer = nn.Identity()

    def forward(self, lrs, tokens, attention_masks, fusion_size=None): 

        edge_index = lrs.edge_index
        # get number of nodes in lrs
        n_nodes = lrs.num_nodes
        batch_size = lrs.num_graphs
        attention_masks = attention_masks
        tokens = tokens
    
        if self.model == 'chemberta-77m-mlm' or self.model == 'chemberta-10m-mlm' or self.model == 'chemberta-77m-mtr' or self.model == 'chemberta-10m-mtr':
            token_embeddings = _chemberta_forward(
                self.chemberta,
                tokens, attention_masks)
            # extract heavy atom embeddings from SMILES token embeddings via custom attention mask
            attention_mask_1d = attention_masks[:,1:-1].reshape(-1)
            atom_embeddings = token_embeddings.reshape(-1, self.embedding_dim)[attention_mask_1d == 1]
        else:
            raise ValueError("invalid SMILES language model")
         # GAT forward
        conv1_out = self.conv1(atom_embeddings, edge_index)
        conv2_out = self.conv2(conv1_out, edge_index)
        conv3_out = self.conv3(conv2_out, edge_index)
        # residual concat
        out = torch.cat((conv1_out, conv2_out, conv3_out), dim=-1)
        out = self.dropout(self.relu(out))  # [n_nodes, 2048]
        # aggregate node vectors to graph
        out = scatter_mean(out, lrs.batch, dim=0)  # [bs, 2048]
        p = self.dense(out).squeeze(-1) + 0.5
        p = self.final_layer(p)
        return p.unsqueeze(1)  # [bs, 1]

class LangGATv2(nn.Module):
    def __init__(self, setup):
        """Language + GATv2 structure fusion. (modified from LM-GVP)


        Args:
            setup : dict, setup file (i.e., setup/setup.json)

        Returns:
            None
        """
        super(LangGATv2, self).__init__()

        self.drop_rate = setup['drop_rate']
        self.seq_len = 160
        self.n_hidden = 512
        self.model = setup['model']
        self.freeze = setup['freeze']
        self.freeze_layer_count = setup['freeze_layer_count']
        self.mode = setup['mode']
        self.num_predictions = setup['num_predictions']

        if setup['model'] == 'chemberta-77m-mlm':
            self.chemberta =  RobertaModel.from_pretrained('config/chembert/ChemBERTa-77M-MLM')
            self.embedding_dim = 384
            _freeze_chemberta(
            self.chemberta,
            freeze_chemberta = setup['freeze'],
            freeze_layer_count = setup['freeze_layer_count'])

        elif setup['model'] == 'chemberta-10m-mlm':
            self.chemberta =  RobertaModel.from_pretrained('config/chembert/ChemBERTa-10M-MLM')
            self.embedding_dim = 384
            _freeze_chemberta(
            self.chemberta,
            freeze_chemberta = setup['freeze'],
            freeze_layer_count = setup['freeze_layer_count'])

        
    
        elif setup['model'] == 'chemberta-77m-mtr':
            self.chemberta =  RobertaModel.from_pretrained('config/chembert/ChemBERTa-77M-MTR')
            self.embedding_dim = 384
            _freeze_chemberta(
            self.chemberta,
            freeze_chemberta = setup['freeze'],
            freeze_layer_count = setup['freeze_layer_count'])

        
        elif setup['model'] == 'chemberta-10m-mtr':
            self.chemberta =  RobertaModel.from_pretrained('config/chembert/ChemBERTa-10M-MTR')
            self.embedding_dim = 384
            _freeze_chemberta(
            self.chemberta,
            freeze_chemberta = setup['freeze'],
            freeze_layer_count = setup['freeze_layer_count'])
        
        elif setup['model'] == 'chemberta-77m-mlm-only' or setup['model'] == 'chemberta-10m-mlm-only' or setup['model'] == 'chemberta-77m-mtr-only' or setup['model'] == 'chemberta-10m-mtr-only' or setup['model'] == 'none':
            raise ValueError('set graph == False for language-only models')
        
        self.conv1 = GATv2Conv(self.embedding_dim, 128, 4)
        self.conv2 = GATv2Conv(512, 128, 4)
        self.conv3 = GATv2Conv(512, 256, 4)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=self.drop_rate)

        self.dense = nn.Sequential(
            nn.Linear(2048, self.n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_hidden, self.num_predictions),
        )

        if self.mode == 'discrete':
            if self.num_predictions == 1:
                self.final_layer = nn.Identity()
                #self.final_layer = nn.Sigmoid()
            elif self.num_predictions > 2:
                self.final_layer = nn.Softmax(dim=1)
            else:
                raise ValueError('invalid number of predictions')
        elif self.mode == 'continuous':
            self.final_layer = nn.Identity()

    def forward(self, lrs, tokens, attention_masks, fusion_size=None): 

        edge_index = lrs.edge_index
        # get number of nodes in lrs
        n_nodes = lrs.num_nodes
        batch_size = lrs.num_graphs
        tokens = tokens
        attention_masks = attention_masks

        if self.model == 'chemberta-77m-mlm' or self.model == 'chemberta-10m-mlm' or self.model == 'chemberta-77m-mtr' or self.model == 'chemberta-10m-mtr':
            token_embeddings = _chemberta_forward(
                self.chemberta,
                tokens, attention_masks)
            # extract heavy atom embeddings from SMILES token embeddings via custom attention mask
            attention_mask_1d = attention_masks[:,1:-1].reshape(-1)
            atom_embeddings = token_embeddings.reshape(-1, self.embedding_dim)[attention_mask_1d == 1]
        else:
            raise ValueError("invalid SMILES language model")
        
         # GATv2 forward
        conv1_out = self.conv1(atom_embeddings, edge_index)
        conv2_out = self.conv2(conv1_out, edge_index)
        conv3_out = self.conv3(conv2_out, edge_index)
        # residual concat
        out = torch.cat((conv1_out, conv2_out, conv3_out), dim=-1)
        out = self.dropout(self.relu(out))  # [n_nodes, 2048]
        # aggregate node vectors to graph
        out = scatter_mean(out, lrs.batch, dim=0)  # [bs, 2048]
        p = self.dense(out).squeeze(-1) + 0.5
        p = self.final_layer(p)
        return p.unsqueeze(1)  # [bs, 1]

    
class LangEnsemble(nn.Module):
    def __init__(self, setup):
        '''
        Args:
            setup : dict, setup file (i.e., setup/setup.json)
        '''
        super(LangEnsemble, self).__init__()
        self.drop_rate = setup['drop_rate']
        self.model = setup['model']
        self.seq_len = 160
        self.fusion_size = 320
        self.mode = setup['mode']
        self.num_predictions = setup['num_predictions']
 
        if setup['model'] == 'chemberta-77m-mlm' or setup['model'] == 'chemberta-77m-mlm-only':
            self.chemberta =  RobertaModel.from_pretrained('config/chembert/ChemBERTa-77M-MLM')
            self.embedding_dim = 384
            _freeze_chemberta(
            self.chemberta,
            freeze_chemberta = setup['freeze'],
            freeze_layer_count = setup['freeze_layer_count'])

        elif setup['model'] == 'chemberta-10m-mlm' or setup['model'] == 'chemberta-10m-mlm-only':
            self.chemberta =  RobertaModel.from_pretrained('config/chembert/ChemBERTa-10M-MLM')
            self.embedding_dim = 384
            _freeze_chemberta(
            self.chemberta,
            freeze_chemberta = setup['freeze'],
            freeze_layer_count = setup['freeze_layer_count'])
    
        elif setup['model'] == 'chemberta-77m-mtr' or setup['model'] == 'chemberta-77m-mtr-only':
            self.chemberta =  RobertaModel.from_pretrained('config/chembert/ChemBERTa-77M-MTR')
            self.embedding_dim = 384
            _freeze_chemberta(
            self.chemberta,
            freeze_chemberta = setup['freeze'],
            freeze_layer_count = setup['freeze_layer_count'])
        
        elif setup['model'] == 'chemberta-10m-mtr' or setup['model'] == 'chemberta-10m-mtr-only':
            self.chemberta =  RobertaModel.from_pretrained('config/chembert/ChemBERTa-10M-MTR')
            self.embedding_dim = 384
            _freeze_chemberta(
            self.chemberta,
            freeze_chemberta = setup['freeze'],
            freeze_layer_count = setup['freeze_layer_count'])

        elif setup['model'] == 'none':
            self.embedding_dim = 1024


        if setup['model'] == 'chemberta-77m-mlm-only' or setup['model'] == 'chemberta-10m-mlm-only' or setup['model'] == 'chemberta-77m-mtr-only' or setup['model'] == 'chemberta-10m-mtr-only':
            self.relu = nn.ReLU(inplace=True)
            self.dropout = nn.Dropout(p=self.drop_rate) 
            self.dense = nn.Sequential(
                nn.Linear(self.fusion_size * self.embedding_dim, self.embedding_dim), 
                nn.ReLU(inplace=True), 
                nn.Linear(self.embedding_dim, self.num_predictions),
            )
        elif setup['model'] == 'chemberta-77m-mlm' or setup['model'] == 'chemberta-10m-mlm' or setup['model'] == 'chemberta-77m-mtr' or setup['model'] == 'chemberta-10m-mtr':
            self.ensdense = nn.Linear(self.fusion_size, self.embedding_dim)
            self.relu = nn.ReLU(inplace=True)
            self.dropout = nn.Dropout(p=self.drop_rate)
            self.dense = nn.Sequential(
            nn.Linear(2 * self.embedding_dim * self.fusion_size, self.embedding_dim), 
            nn.ReLU(inplace=True), 
            nn.Linear(self.embedding_dim, self.num_predictions),
            ) 
        elif setup['model'] == 'none':
            self.ensdense = nn.Linear(self.fusion_size, self.embedding_dim)
            self.relu = nn.ReLU(inplace=True)
            self.dropout = nn.Dropout(p=self.drop_rate)
            self.dense = nn.Sequential(
            nn.Linear(self.fusion_size * self.embedding_dim, self.embedding_dim), 
            nn.ReLU(inplace=True), 
            nn.Linear(self.embedding_dim, self.num_predictions),
            )
        
        if self.mode == 'discrete':
            if self.num_predictions == 1:
                self.final_layer = nn.Identity()
                #self.final_layer = nn.Sigmoid()
            elif self.num_predictions > 2:
                self.final_layer = nn.Softmax(dim=1)
            else:
                raise ValueError('invalid number of predictions')
        elif self.mode == 'continuous':
            self.final_layer = nn.Identity()
        
    def forward(self, srs, tokens, attention_masks, fusion_size=None):                                  
        '''
        Combines ensemble fusion with atom embeddings as an input                                                                                                   tensor x.
        Args:
            x : tensor (B, SRS, tokens), input ensemble fusion & SMILES tokens
        Returns:
            out: tensor (B, PROP), hidden states
        
        '''
        # ensemble fusion --> hidden layer 
        if srs is not None:
            ensemble_fusions = srs
            batch_size, heigth, width = ensemble_fusions.shape
        else:
            batch_size, length = tokens.shape

        # push sequences
        tokens = tokens
        attention_masks = attention_masks

        if self.model == 'chemberta-77m-mlm' or self.model == 'chemberta-10m-mlm' or self.model == 'chemberta-77m-mtr' or self.model == 'chemberta-10m-mtr' or self.model == 'chemberta-77m-mlm-only' or self.model == 'chemberta-10m-mlm-only' or self.model == 'chemberta-77m-mtr-only' or self.model == 'chemberta-10m-mtr-only':
            token_embeddings = _chemberta_forward(
                self.chemberta,
                tokens, attention_masks)
            
            attention_mask = attention_masks[:,1:-1]
            attention_mask_1d = attention_mask[:,1:-1].reshape(-1)
            atom_embeddings = token_embeddings.reshape(-1, self.embedding_dim)[attention_mask_1d == 1]
            
            if self.model == 'chemberta-77m-mlm' or self.model == 'chemberta-10m-mlm' or self.model == 'chemberta-77m-mtr' or self.model == 'chemberta-10m-mtr':
                srs_embeddings = self.ensdense(ensemble_fusions)
                ens_lang = torch.cat((srs_embeddings, atom_embeddings), dim =2) # (B, tokens = 320 (false), LANG_EMBEDDING + SRS_EMBEDDING)
                ens_lang = ens_lang.view(batch_size, -1) # (B, [LCSEQ + HCSEQ]*[LANG_EMBEDDING + SRS_EMBEDDING])

            elif self.model == 'chemberta-77m-mlm-only' or self.model == 'chemberta-10m-mlm-only' or self.model == 'chemberta-77m-mtr-only' or self.model == 'chemberta-10m-mtr-only':
                ens_lang = token_embeddings.view(batch_size, -1)
            else:
                raise ValueError("invalid language model")
            
        elif self.model == 'none':
            srs_embeddings = self.ensdense(ensemble_fusions)
            ens_lang = srs_embeddings.view(batch_size, -1) # (B, [SRS_EMBEDDING] = 245760)

        p = self.relu (ens_lang)
        p = self.dropout (p)
        p = self.dense(p)
        p = self.final_layer(p)

        return p

class SLEFNet(nn.Module):
    ''' Small molecule Language Ensemble Fusion, a neural network for recursive fusion of small molecule structural ensembles & language.  '''

    def __init__(self, setup):
        '''
        Args:
            setup : dict, setupuration file
        '''

        super(SLEFNet, self).__init__()
        if setup["language"]["graph"] == False:
            self.encode = Encoder(setup["encoder"])
            self.superres = SuperResTransformer(dim=setup["encoder"]["n_hidden_fusion"],
                                                depth=setup["transformer"]["depth"],
                                                heads=setup["transformer"]["heads"],
                                                mlp_dim=setup["transformer"]["mlp_dim"],
                                                dropout=setup["transformer"]["dropout"])
            self.decode = Decoder(setup)
            self.property = LangEnsemble(setup["language"])
            self.GCNproperty = None
            self.GATproperty = None
            self.GATv2property = None
        elif setup["language"]["graph"] == "GCN":
            self.GCNproperty = LangGCN(setup["language"])
            self.GATproperty = None
            self.GATv2property = None
        elif setup["language"]["graph"] == "GAT":
            self.GCNproperty = None
            self.GATproperty = LangGAT(setup["language"])
            self.GATv2property = None
        elif setup["language"]["graph"] == "GATv2":
            self.GCNproperty = None
            self.GATv2property = LangGATv2(setup["language"])
            self.GATproperty = None
        else:
            raise ValueError("graph must be either 'GCN', 'GAT', or 'GATv2' or false in setup.json")


    def forward(self, lrs, ens_attention_masks, alphas, tokens, attention_masks, fusion_size):
        '''
        Super resolves a batch of low-resolution ensembles, integrates pretrained language model, and predicts small molecule property.
        Args:
            lrs : tensor (B, C, L, W, H), low-resolution ensemble
            ens_attention_masks: tensor (B, L, W, H), ensemble attention masks for CNN transformer
            alphas : tensor (B, L), boolean indicator (0 if collated ensemble, 1 otherwise)
            tokens: tensor (B, L), tokenized SMILES length (ProtBERT convention)
            attention_masks: tensor (B, L) tokenized attention masks for SMILES heavy atoms
        Returns:
            srs: tensor (B, C_out, W, H), super-resolved ensemble
            prop: tensor(B, PROP), predicted property of antibody
        '''

        if self.GATproperty is not None:
            prop = self.GATproperty(lrs, tokens, attention_masks, fusion_size=fusion_size)
        elif self.GATv2property is not None:
            prop = self.GATv2property(lrs, tokens, attention_masks, fusion_size=fusion_size)
        elif self.GCNproperty is not None:
            prop = self.GCNproperty(lrs, tokens, attention_masks, fusion_size=fusion_size)
        elif lrs is not None and self.GCNproperty is None and self.GATproperty is None and self.GATv2property is None:
            batch_size, channel_len, seq_len, heigth, width = lrs.shape
            #batch_size, seq_len, heigth, width = lrs.shape
            stacked_input = lrs.view(batch_size * seq_len, channel_len, width, heigth)
            layer1 = self.encode(stacked_input) # encode input tensor

            ####################### encode ensemble ######################
            layer1 = layer1.view(batch_size, seq_len, -1, width, heigth) # tensor (B, L, C, W, H)
            #print('encode, tensor (B, L, C, W, H)')
            #print(layer1.shape)
            ####################### fuse ensemble ######################
            img = layer1.permute(0, 3, 4, 1, 2).reshape(-1, layer1.shape[1], layer1.shape[2])  # .contiguous().view == .reshape()
            # print('permute/reshape encode')
            if ens_attention_masks is not None:
                ens_attention_masks = ens_attention_masks.permute(0, 2, 3, 1).reshape(-1, ens_attention_masks.shape[1])
            # print(ens_attention_masks.shape)
            preds = self.superres(img, mask=alphas, maps=ens_attention_masks, fusion_size=fusion_size)
            #print(fusion_size)
            #print(preds.shape)
            # preds = preds.reshape(-1, 320, 320, preds.shape[-1]).permute(0, 3, 1, 2)
            preds = preds.view(-1, width, heigth, preds.shape[-1]).permute(0, 3, 1, 2)
            #print('fuse, tensor (?)')
            #print(preds.shape)
            ####################### decode ensemble ######################
            srs = self.decode(preds)  # decode final hidden state (B, C_out, W, H)
            srs = srs.squeeze(1)
            #print('srs, tensor (B, C_out, W, H)')
            #print(srs.shape)
            ################# predict property (ensemble + language embeddings) ############
            prop = self.property(srs, tokens, attention_masks, fusion_size=fusion_size)
        else: 
            srs = None
            prop = self.property(srs, tokens, attention_masks, fusion_size=fusion_size)
        return prop
