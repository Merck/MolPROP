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

import os
import time
import random
import argparse
import torch
import pandas as pd
from tokenizers import Tokenizer
from transformers import BertModel
from transformers.models.roberta import RobertaModel, RobertaTokenizer
import pandas as pd
import numpy as np
import deepchem as dc
import rdkit
import os
import argparse


if __name__ == '__main__':
    os.chdir('pathway/to/data')

    # arg for data
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='esolv')
    parser.add_argument('--X', type=str, default='Compound ID')
    parser.add_argument('--y', type=str, default='measured log solubility in mols per litre')
    parser.add_argument('--ids', type=str, default='smiles')
    parser.add_argument('--frac_train', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=123)

    args = parser.parse_args()

    data = pd.read_csv(args.data + '.csv')
    print(data.head())

    dataset = dc.data.DiskDataset.from_numpy(X=data[args.X], y=data[args.y],ids=data[args.ids])
    scaffoldsplitter = dc.splits.ScaffoldSplitter()
    train, holdout = scaffoldsplitter.train_test_split(dataset, frac_train=args.frac_train, seed=args.seed)
    train= train.to_dataframe()
    holdout = holdout.to_dataframe()
    train.to_csv("train_" + args.data + ".csv")
    holdout.to_csv("holdout_" + args.data + ".csv")
