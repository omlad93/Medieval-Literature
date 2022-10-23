import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent.parent))

from parsing.parse_csv import *
from Labels.labels import *
from model.ttm_dataset import *
from tqdm import tqdm
from sklearn import metrics, utils
from transformers import DistilBertForTokenClassification, RobertaForTokenClassification
import torch
from torch import cuda
from torch.utils.data import DataLoader
from functools import cmp_to_key

REPO_FOLDER = str(Path(__file__).parent.parent.parent)
DEVICE = 'cuda' if cuda.is_available() else 'cpu'

LEARNING_RATE = 5e-5
EPOCHS = 25
model: DistilBertForTokenClassification
loss_func: torch.nn.CrossEntropyLoss
optimizer: torch.optim.Adam
num_labels: int

def loader(df: DataFrame, default_label, tag2idx):
    print(f"Total Fragments: {len(df.index)}")
    ''' Prepare Data '''
    f = 0.2
    tst_data = df.sample(frac=f, random_state=200)
    trn_data = df.drop(tst_data.index).reset_index(drop=True)
    tst_data = tst_data.reset_index(drop=True)
    trn_set = TTMDataset(trn_data, default_label, tag2idx)
    tst_set = TTMDataset(tst_data, default_label, tag2idx)
    trn_labels = np.array(trn_set.labels).flatten()
    loader_params = {'batch_size': BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }
    trn_loader = DataLoader(trn_set, **loader_params)
    tst_loader = DataLoader(tst_set, **loader_params)
    return (trn_loader, trn_set.unique_labels, trn_labels, tst_loader, tst_set.unique_labels)

def train_epoch(trn_loader):
    ''' Train Data'''
    total_loss = 0
    steps = 0
    gradient_acc_steps = EFFECTIVE_BATCH_SIZE/BATCH_SIZE
    model.train()
    optimizer.zero_grad()
    for tokens, labels in tqdm(trn_loader):
        labels = labels.to(DEVICE)
        # squeeze in order to match the sizes. From [batch,1,seq_len] --> [batch,seq_len] 
        mask = tokens['attention_mask'].squeeze(1).to(DEVICE)
        ids = tokens['input_ids'].squeeze(1).to(DEVICE)

        logits = model(ids, mask).logits
        active_loss = mask.view(-1) == 1
        active_logits = logits.view(-1, num_labels)
        active_labels = torch.where(
            active_loss, labels.view(-1), torch.tensor(loss_func.ignore_index).type_as(labels)
        )
        loss = loss_func(active_logits, active_labels)
        total_loss += loss.item()
        loss /= gradient_acc_steps
        loss.backward()
        if (steps+1) % gradient_acc_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
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
    return accuracy, f1, fin_targets, fin_preds

def draw_confusion_matrix(y_true, y_preds, sorted_labels):
    fig, ax = plt.subplots(figsize=(16,16))
    ax.tick_params(axis='both', labelsize=4)
    metrics.ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_preds,
        normalize='true',
        xticks_rotation='vertical',
        include_values=False,
        cmap=plt.cm.Blues,
        ax=ax)
    plt.savefig('all_labels.png')
    ax.cla()
    ax.tick_params(axis='both', labelsize=10)
    metrics.ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_preds,
        labels=sorted_labels[-15:],
        normalize='true',
        xticks_rotation='vertical',
        colorbar=False,
        cmap=plt.cm.Blues,
        ax=ax)
    plt.savefig('top_labels.png')
    ax.cla()
    metrics.ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_preds,
        labels=sorted_labels[:14] + ["0"],
        normalize='true',
        xticks_rotation='vertical',
        colorbar=False,
        cmap=plt.cm.Blues,
        ax=ax)
    plt.savefig('bottom_labels.png')

def set_globals(local_num_labels, trn_targets):
    global model, loss_func, optimizer, num_labels
    num_labels = local_num_labels
    model = RobertaForTokenClassification.from_pretrained(f"{REPO_FOLDER}/Code/model/pretrained", num_labels=num_labels)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    # class_weights = utils.class_weight.compute_class_weight('balanced', classes=np.unique(trn_targets), y=trn_targets)
    # loss_func = torch.nn.CrossEntropyLoss(torch.tensor(class_weights, dtype=torch.float).to(DEVICE))
    loss_func = torch.nn.CrossEntropyLoss()


def main():
    print("hi")
    save_model_path = f"{REPO_FOLDER}/trained_models/tc_pretrained_corrections_roberta.pt"
    original_df = parse_csv_token_classification(f"{REPO_FOLDER}/Data/csv/per-word-combined.csv", fix_spelling=True)
    tag2idx, idx2tag, default_label, unique_tags = tags_mapping(original_df["labels"])
    # df = filter_ignored_labels(original_df, unique_tags.keys())
    trn_loader, trn_labels, trn_targets, tst_loader, tst_labels = loader(original_df, default_label, tag2idx)
    trn_labels = sorted([idx2tag[x] for x in trn_labels], key=cmp_to_key(lambda a, b: unique_tags[a] - unique_tags[b]))
    tst_labels = sorted([idx2tag[x] for x in tst_labels], key=cmp_to_key(lambda a, b: unique_tags[a] - unique_tags[b]))
    
    set_globals(len(unique_tags), trn_targets)
    best_f1, best_acc = 0, 0
    for epoch in range(EPOCHS):
        loss = train_epoch(trn_loader)
        print(f"Finished Epoch: {epoch+1}, Loss: {loss}")
        acc, f1, y_true, y_pred = evaluation(tst_loader)
        print(f"Test Accuracy: {acc}, F1: {f1}")
        if f1 > best_f1 and (acc > best_acc or best_acc - acc < 0.01):
            best_f1 = f1
            best_acc = acc
            torch.save(model.state_dict(), save_model_path)
        if epoch == EPOCHS - 1:
            model.load_state_dict(torch.load(save_model_path))
            model.eval()
            draw_confusion_matrix([idx2tag[x] for x in y_true], [idx2tag[x] for x in y_pred], tst_labels)

if __name__ == "__main__":
    main()
