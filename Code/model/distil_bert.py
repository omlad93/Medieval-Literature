import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent.parent))
from parsing.parse_csv import *
from Labels.labels import *
from transformers import DistilBertTokenizer, DistilBertModel
import torch
from torch.utils.data import Dataset
import logging
logging.basicConfig(level=logging.ERROR)

MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1e-05

class MultiLabelDataset(Dataset):
    
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer: DistilBertTokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.labels
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }


class DistilBERT(torch.nn.Module):
    def __init__(self):
        super(DistilBERT, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, 104)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

def loss_func(outputs, targets):
    loss = torch.nn.BCEWithLogitsLoss()
    return loss(outputs, targets)