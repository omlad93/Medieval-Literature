import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent.parent))
from parsing.parse_csv import *
from Labels.labels import *
from transformers import DistilBertModel
import torch

MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-05

class DistilBERTTutorial(torch.nn.Module):
    def __init__(self):
        super(DistilBERTTutorial, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, 2)

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