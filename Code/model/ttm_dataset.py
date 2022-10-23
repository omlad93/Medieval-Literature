import numpy as np
import pandas as pd
from transformers import AutoTokenizer, PreTrainedTokenizer
import torch
from torch.utils.data import Dataset
from matplotlib import pyplot as plt

MAX_LEN = 256
BATCH_SIZE = 4
EFFECTIVE_BATCH_SIZE = 8
INSTANCE_THRESHOLD = 0

tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)

def draw_charts(data: dict[str, int], histogram=True):
  labels = []
  instances = []
  for key, value in data.items():
    if key != "0":
      labels.append(key)
      instances.append(value)
  if histogram:
    bins = np.arange(0, np.ceil(max(instances) / 10.0) * 10 + 10, 10)
    plt.hist(instances, bins=bins, edgecolor='black', linewidth=1)
    plt.xticks(np.arange(0, bins[-1], 50))
    plt.xlim(left=0)
    plt.title('Labels Frequency Distribution')
    plt.xlabel('Instances')
    plt.ylabel('No. of labels')
  else:
    plt.rcdefaults()
    fig, ax = plt.subplots()
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, instances, align='center', color='maroon')
    ax.margins(y=0.01)
    ax.set_yticks(y_pos, labels=labels, fontsize=3)
    ax.invert_yaxis() # labels read top-to-bottom
    # Add annotation to bars
    for i in ax.patches:
      plt.text(i.get_width()+0.2, i.get_y()+0.8, str(i.get_width()), fontsize=3)
    ax.set_xlabel('No. of instances in data')
    ax.set_title('Labels in Dataset')
  plt.show()

def tags_mapping(tags_series: pd.Series, threshold=INSTANCE_THRESHOLD, show_charts=False):
  """
  tag_series = df column with tags for each sentence.
  Returns:
    - dictionary mapping tags to indices (label)
    - dictionary mapping indices to tags
    - The label corresponding to tag '0'
    - A set of unique tags ecountered in the trainind df, this will define the classifier dimension
  """
  unique_tags = {}
  
  for tag_mask in tags_series.to_list():
    for tag in tag_mask.split():
      if tag in unique_tags:
        unique_tags[tag] += 1
      else:
        unique_tags[tag] = 1
  
  tags_to_remove = []
  for tag, instances in unique_tags.items():
    if instances < threshold:
      tags_to_remove.append(tag)
  for tag in tags_to_remove:
    unique_tags.pop(tag)

  tag2idx = {k:v for v,k in enumerate(sorted(unique_tags.keys()))}
  idx2tag = {k:v for v,k in tag2idx.items()}
  unseen_label = tag2idx["0"]

  if show_charts:
    draw_charts(unique_tags, False)

  return tag2idx, idx2tag, unseen_label, unique_tags

def match_tokens_labels(tokenized_input, tags, tag2idx, ignore_token = 0):
    word_ids = tokenized_input.word_ids()
    label_ids = []
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(ignore_token)
        #if its equal to the previous word we can add the same label id of the previous or ignore_token 
        else:
            try:
                reference_tag = tags[word_idx]
                label_ids.append(tag2idx[reference_tag])
            except:
                label_ids.append(ignore_token)
    return label_ids

def decode_tokens(input_ids):
  return [' '.join(tokenizer.convert_ids_to_tokens(x)) for x in input_ids]

class TCPredictionDataset(Dataset):
  """
  Custom dataset implementation to get only text batches for predictions
  Inputs:
   - df : dataframe with column [text]
  """
  
  def __init__(self, df):
    if not isinstance(df, pd.DataFrame):
      raise TypeError('Input should be a dataframe')
    
    if "text" not in df.columns:
      raise ValueError("Dataframe should contain 'text' column")

    texts = df["text"].values.tolist()
    self.texts = [tokenizer(text, padding = "max_length", max_length=MAX_LEN, truncation = True, return_tensors = "pt") for text in texts]

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, idx):
    batch_text = self.texts[idx]
    return batch_text

class TTMDataset(Dataset):
  """
  Custom dataset implementation to get (text,labels) tuples
  Inputs:
   - df : dataframe with columns [text, labels]
  """
  
  def __init__(self, df, default_label, tag2idx):
    if not isinstance(df, pd.DataFrame):
      raise TypeError('Input should be a dataframe')
    
    if "labels" not in df.columns or "text" not in df.columns:
      raise ValueError("Dataframe should contain 'text' and 'labels' columns")

    tags_list = [i.split() for i in df["labels"].values.tolist()]
    texts = df["text"].values.tolist()
    self.texts = [tokenizer(text, padding = "max_length", max_length=MAX_LEN, truncation = True, return_tensors = "pt") for text in texts]
    self.labels = [match_tokens_labels(text, tags, tag2idx, default_label) for text,tags in zip(self.texts, tags_list)]
    self.unique_labels = set(np.array(self.labels).flatten())

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    batch_text = self.texts[idx]
    batch_labels = self.labels[idx]
    return batch_text, torch.LongTensor(batch_labels)