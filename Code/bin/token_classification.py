import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent.parent))

from parsing.parse_csv import *
from Labels.labels import *
from model.ttm_dataset import *
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from transformers import DistilBertForTokenClassification
import torch
from torch import cuda
from torch.utils.data import DataLoader
import logging
logging.basicConfig(level=logging.ERROR)

REPO_FOLDER = str(Path(__file__).parent.parent.parent)
DEVICE = 'cuda' if cuda.is_available() else 'cpu'

LEARNING_RATE = 5e-5
EPOCHS = 10
model: DistilBertForTokenClassification
optimizer: torch.optim.Adam

def init_dataframe():
    my_df = parse_csv_token_classification(f"{REPO_FOLDER}/Data/csv/per-word-combined.csv")
    # print(my_df.head())
    return my_df

def loader(df: DataFrame, default_label, tag2idx):
    print(len(df.index))
    ''' Prepare Data '''
    f = 0.2
    tst_data = df.sample(frac=f, random_state=200).reset_index(drop=True)
    trn_data = df.drop(tst_data.index).reset_index(drop=True)
    trn_set = TTMDataset(trn_data, default_label, tag2idx)
    tst_set = TTMDataset(tst_data, default_label, tag2idx)
    loader_params = {'batch_size': BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }
    trn_loader = DataLoader(trn_set, **loader_params)
    tst_loader = DataLoader(tst_set, **loader_params)
    return (trn_loader,tst_loader)

def train_epoch(trn_loader):
    ''' Train Data'''
    total_loss = 0
    steps = 0
    loss_func = torch.nn.CrossEntropyLoss()
    model.train()
    for tokens, labels in tqdm(trn_loader):
        labels = labels.to(DEVICE)
        # squeeze in order to match the sizes. From [batch,1,seq_len] --> [batch,seq_len] 
        mask = tokens['attention_mask'].squeeze(1).to(DEVICE)
        ids = tokens['input_ids'].squeeze(1).to(DEVICE)

        optimizer.zero_grad()
        logits = model(ids, mask).logits
        active_loss = mask.view(-1) == 1
        active_logits = logits.view(-1, 104)
        active_labels = torch.where(
            active_loss, labels.view(-1), torch.tensor(loss_func.ignore_index).type_as(labels)
        )
        loss = loss_func(active_logits, active_labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        steps += 1
    return total_loss / steps

def evaluation(testing_loader):
    model.eval()
    fin_targets=[]
    fin_preds=[]
    with torch.no_grad():
        for tokens, labels in tqdm(testing_loader):
            labels = labels.to(DEVICE)
            # squeeze in order to match the sizes. From [batch,1,seq_len] --> [batch,seq_len] 
            mask = tokens['attention_mask'].squeeze(1).to(DEVICE)
            ids = tokens['input_ids'].squeeze(1).to(DEVICE)
            logits = model(ids, mask).logits
            predictions = logits.argmax(dim=-1)
            active = (mask.view(-1) == 1).to("cpu").numpy()
            fin_targets.extend(labels.cpu().detach().numpy().flatten()[active])
            fin_preds.extend(predictions.to("cpu").numpy().flatten()[active])
    accuracy = metrics.accuracy_score(fin_targets, fin_preds)
    f1 = metrics.f1_score(fin_targets, fin_preds, average='weighted')
    return accuracy, f1

def main():
    print("hi")
    df = init_dataframe()
    tag2idx, idx2tag, default_label, unique_tags = tags_mapping(df["labels"], 0)
    trn_loader, tst_loader = loader(df, default_label, tag2idx)
    global model, optimizer
    model = DistilBertForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels = len(unique_tags))
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    for epoch in range(EPOCHS):
        loss = train_epoch(trn_loader)
        print(f"Finished Epoch: {epoch+1}, Loss: {loss}")
        results = evaluation(tst_loader)
        print(f"Test Accuracy: {results[0]}, F1: {results[1]}")

if __name__ == "__main__":
    main()
