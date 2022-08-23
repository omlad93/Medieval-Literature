import sys
from typing import Optional, Sequence
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent.parent))

from parsing.parse_csv import *
from Labels.labels import *
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
MODEL = DistilBERT()
MODEL.to(DEVICE)
OPTIMIZER = torch.optim.Adam(params = MODEL.parameters(), lr=LEARNING_RATE)

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

def init_dual_dataframe():
    ''' init Dataframes (1 for our case, one for example case''' 
    # The CSV from the online example had a column for each possible label
    # csv_df = pd.read_csv("./ETC/train.csv",encoding='unicode_escape', keep_default_na=False)
    # csv_df.drop(["id"], inplace=True,axis=1)
    my_df = parse_single_csv(f"{REPO_FOLDER}/Data/csv/Macbeth.csv",slim=True)
    my_df["Topics"]=my_df["Topics"].apply(labels_as_boolean)
    my_df.rename(columns={
        "Fragment":"text",
        "Topics":"labels"
        }, inplace=True
    )
    # template = pd.DataFrame()
    # template['text'] = csv_df['comment_text']
    # template['labels'] = csv_df.iloc[:, 1:].values.tolist()
    # return template,my_df
    return my_df

            #  L1 L2 
#  Fragment : [0  1   0 1 0 0 0 0 0 ... 0 1 ]

def loader(df: DataFrame):
    print(len(df.index))
    ''' Prepare Data '''
    f = 0.1
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
    # s,r = 0,0
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
    # print(f"Finished with Ratio = {100*r/s}%")

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

def main():
    print("hi")
    df = init_dual_dataframe()
    # for i,df in enumerate(dfs):
    trn_loader, tst_loader = loader(df)
    for epoch in range(EPOCHS):
        loss = train_epoch(trn_loader)
        print(f"Finished Epoch: {epoch+1}, Loss: {loss}")
    results = evaluation(tst_loader)
    print(f"Test Hamming Score: {results[0]}, Hamming Loss: {results[1]}, F1: {results[2]}")

if __name__ == "__main__":
    main()
