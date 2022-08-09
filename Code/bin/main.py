
import sys
from typing import Optional, Sequence
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent.parent))

from parsing.parse_csv import *
from labels.labels import *
from model.distil_bert import *
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
import transformers
from transformers import DistilBertTokenizer, DistilBertModel
import torch
from torch import cuda
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import logging
logging.basicConfig(level=logging.ERROR)

# DEVICE = 'cuda' if cuda.is_available else 'CPU'
DEVICE='cpu'
MODEL = DistilBERT()
MODEL.to(DEVICE)
OPTIMIZER = torch.optim.Adam(params = MODEL.parameters(), lr=LEARNING_RATE)

def init_dual_dataframe():
    # The CSV from the online example had a column for each possible label
    csv_df = pd.read_csv("./ETC/train.csv",encoding='unicode_escape', keep_default_na=False)
    csv_df.drop(["id"], inplace=True,axis=1)
    my_df = parse_single_csv("./data/csv/Macbeth.csv",slim=True)
    my_df["Topics"]=my_df["Topics"].apply(labels_as_boolean)
    my_df.rename(columns={
        "Fragment":"text",
        "Topics":"labels"
        }, inplace=True
    )
    template = pd.DataFrame()
    template['text'] = csv_df['comment_text']
    template['labels'] = csv_df.iloc[:, 1:].values.tolist()
    return template,my_df


def loader(df):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', truncation=True, do_lower_case=True)
    f = 0.8
    trn_data = df.sample(frac=f, random_state=200)
    tst_data = df.drop(trn_data.index).reset_index(drop=True)
    trn_set = MultiLabelDataset(trn_data,tokenizer,MAX_LEN)
    tst_set = MultiLabelDataset(tst_data,tokenizer,MAX_LEN)
    loader_params = {'batch_size': BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }
    trn_loader = DataLoader(trn_set, **loader_params)
    tst_loader = DataLoader(tst_set, **loader_params)
    return (trn_loader,tst_loader)

def train(trn_loader):
    s,r = 0,0
    MODEL.train()
    for i,(_, data) in enumerate(tqdm(enumerate(trn_loader,0))):
            ids = data['ids'].to(DEVICE, dtype = torch.long)
            mask = data['mask'].to(DEVICE, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(DEVICE, dtype = torch.long)
            targets = data['targets'].to(DEVICE, dtype = torch.float)

            ret = MODEL(ids,mask,token_type_ids)
            OPTIMIZER.zero_grad()
            loss=loss_func(ret,targets)

            loss.backward()
            OPTIMIZER.step()
    print(f"Finished with Ratio = {100*r/s}%")

def convert_dataset_to_tensor(dataset: tuple):
    inputs = torch.tensor(dataset[0])
    masks = torch.tensor(dataset[1])
    labels = torch.tensor(dataset[2])
    return inputs, masks, labels


def main():
    _, df = init_dual_dataframe()
    # for i,df in enumerate(dfs):
    trn_loader, tst_loader = loader(df)
    convert_dataset_to_tensor(trn_loader)
    # train(trn_loader)
    print(f"Finished {i}")



if __name__ == "__main__":
    main()
