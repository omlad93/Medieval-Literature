import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent.parent))

from parsing.parse_csv import *
from Labels.labels import *
from model.ttm_dataset import *
from tqdm import tqdm
from transformers import RobertaForTokenClassification
import torch
from torch import cuda
from torch.utils.data import DataLoader

REPO_FOLDER = str(Path(__file__).parent.parent.parent)
DEVICE = 'cuda' if cuda.is_available() else 'cpu'

model: RobertaForTokenClassification
num_labels: int

def loader(df: DataFrame):
    print(f"Total Fragments: {len(df.index)}")
    ''' Prepare Data '''
    dataset = TCPredictionDataset(df)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    return loader

def prediction(loader):
    model.eval()
    fin_preds=[]
    fin_tokenized_texts = []
    with torch.no_grad():
        for tokens in tqdm(loader):
            # squeeze in order to match the sizes. From [batch,1,seq_len] --> [batch,seq_len] 
            mask = tokens['attention_mask'].squeeze(1).to(DEVICE)
            active = (mask == 1).to("cpu").numpy()
            ids = tokens['input_ids'].squeeze(1).to(DEVICE)
            tokenized_sentences = decode_tokens(ids)
            tokenized_sentences = [text[:text.index(" <pad>")].replace('Ġ', '') for text in tokenized_sentences]
            fin_tokenized_texts.extend(tokenized_sentences)
            logits = model(ids, mask).logits
            predictions = logits.argmax(dim=-1)
            predictions = [predictions.to("cpu").numpy()[i, active[i]] for i in range(len(predictions))]
            fin_preds.extend(predictions)
    return fin_tokenized_texts, fin_preds

def set_globals(local_num_labels):
    global model, num_labels
    num_labels = local_num_labels
    model = RobertaForTokenClassification.from_pretrained("roberta-base", num_labels=num_labels)
    model.load_state_dict(torch.load(f"{REPO_FOLDER}/trained_models/tc_pretrained_corrections_roberta.pt", map_location=DEVICE))
    model.to(DEVICE)

def main():
    data_file_path = sys.argv[1] # f"{REPO_FOLDER}/Data/csv/sample-test-data.csv"
    df = parse_csv_for_prediction(data_file_path, fix_spelling=True)
    tags_file = open(f"{REPO_FOLDER}/Code/Labels/unique_tags_dict.pkl", "rb")
    unique_tags: dict = pickle.load(tags_file)
    tags_file.close()
    tag2idx = {k:v for v,k in enumerate(sorted(unique_tags.keys()))}
    idx2tag = {k:v for v,k in tag2idx.items()}
    dataloader = loader(df)
    set_globals(len(unique_tags))
    tokens, preds = prediction(dataloader)
    preds = [' '.join([idx2tag[idx] for idx in x]) for x in preds]
    df['tokens'] = tokens
    df['predictions'] = preds
    df.to_csv('output.csv')

if __name__ == "__main__":
    main()
