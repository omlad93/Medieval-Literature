from lib2to3.pgen2.token import OP
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

REPO_FOLDER = str(Path(__file__).parent.parent.parent)
DEVICE = 'cuda' if cuda.is_available() else 'cpu'
MODEL       = None
OPTIMIZER   = None

def calc_hamming_score(y_true, y_pred):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)
    return np.mean(acc_list)

def init_data(labels_limit=None,fix_spelling:bool=False):
    global MODEL
    global OPTIMIZER
    ''' init Dataframes (1 labels keywords , one for plays)''' 
    # labeled_plays = parse_single_csv(f"{REPO_FOLDER}/data/csv/combined.csv",slim=True)
    dfs,_ = zip(*parse_all_csv_in_directory("data\csv",fix_spelling=fix_spelling,save=False))
    labeled_plays_df = pd.concat([df for df in dfs if is_labeled(df)], axis=0, ignore_index=True)
    options = get_top_labels(labeled_plays_df,labels_limit)
    classes = len(options) if labels_limit else len(Label)
    labeled_plays_df["labels"]=labeled_plays_df["labels"].apply(lambda x: labels_as_boolean(x,options))
    labeled_words_df = labeled_words(options)
    # Model Init
    MODEL = DistilBERT(classes=classes)
    MODEL.to(DEVICE)
    OPTIMIZER = torch.optim.Adam(params = MODEL.parameters(), lr=LEARNING_RATE)
    return labeled_words_df,labeled_plays_df


def loader(df: DataFrame, f:int=0.1):
    # print(len(df.index))
    ''' Prepare Data '''
    tst_data = df.sample(frac=f, random_state=200).reset_index(drop=True)
    trn_data = df.drop(tst_data.index).reset_index(drop=True)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', truncation=True, do_lower_case=True)
    trn_set = MultiLabelDataset(trn_data,tokenizer,MAX_LEN)
    tst_set = MultiLabelDataset(tst_data,tokenizer,MAX_LEN)
    loader_params = {'batch_size': BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }
    trn_loader = DataLoader(trn_set, **loader_params)
    tst_loader = DataLoader(tst_set, **loader_params)
    return (trn_loader,tst_loader)

def train_epoch(trn_loader):
    ''' Train Data'''
    loss = 0
    MODEL.train()
    for _, data in tqdm(enumerate(trn_loader,0)):
        ids = data['ids'].to(DEVICE, dtype = torch.long)
        mask = data['mask'].to(DEVICE, dtype = torch.long)
        targets = data['targets'].to(DEVICE, dtype = torch.float)

        ret = MODEL(ids,mask)
        OPTIMIZER.zero_grad()
        loss=loss_func(ret,targets)

        loss.backward()
        OPTIMIZER.step()
    return loss.item()

def evaluation(testing_loader):
    MODEL.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            ids = data['ids'].to(DEVICE, dtype = torch.long)
            mask = data['mask'].to(DEVICE, dtype = torch.long)
            targets = data['targets'].to(DEVICE, dtype = torch.float)
            outputs = MODEL(ids, mask)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    fin_preds = (np.array(fin_outputs, dtype='f') >= 0.5).astype(float)
    fin_targets = np.array(fin_targets)
    hamming_loss = metrics.hamming_loss(fin_targets, fin_preds)
    hamming_score = calc_hamming_score(fin_targets, fin_preds)
    f1 = metrics.f1_score(fin_targets, fin_preds, average='micro')
    return hamming_loss, hamming_score, f1


def get_top_labels(df:pd.DataFrame,top:int=10):
    if top:
        rates = [0 for label in Label]
        for current_labeling in df.labels:
            rates = [ ri+ci for ri,ci in zip(rates, labels_as_int(current_labeling)) ]
        rated_labels = {label:rates[i] for i,label in enumerate(Label) }
        top_labels = {
            label:rated_labels[label]
            for label in sorted(rated_labels.keys(), key=lambda x:-rated_labels[x])[:top]
        }
        return list(top_labels.keys())
    return Label
    

def main():
    dfs = init_data(labels_limit=10,fix_spelling=True)
    # dfs = init_data() # NO LIMIT
    for i,df in enumerate(dfs):
        trn_loader, tst_loader = loader(df, f=0.1 if i else 1/len(df))
        for epoch in range(EPOCHS):
            loss = train_epoch(trn_loader)
            print(f"Finished Epoch: {epoch+1}, Loss: {loss}")
        results = evaluation(tst_loader)
    print(f"Test Hamming Score: {results[0]}, Hamming Loss: {results[1]}, F1: {results[2]}")

if __name__ == "__main__":
    main()
