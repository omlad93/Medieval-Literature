from __future__ import annotations
import re
import sys
from pathlib import Path
from typing import Sequence
sys.path.insert(0,str(Path(__file__).parent.parent))
import os
import pandas as pd # type: ignore
from pandas import DataFrame,Series
import pickle
from Labels.labels import Label, words_dict
from parsing.word_corrections import rephrase

MISSING_LABELS :set[str] = set()

def convert_labels(topic_list: list[str])->list[Label]:
    '''
    A function to make Labels from Topics (strings)
    '''
    for i,topic in enumerate(topic_list):
        label = Label.get_label(topic)
        if label == Label.NONE:
            MISSING_LABELS.add(topic)
        topic_list[i] = label
    return topic_list
   
def combine_topics(x:Series) -> list[Label]:
    '''
    A function to combine all Topics columns into a single column
    '''
    topics = [c for c in x.index if 'Topic' in c]
    return convert_labels([x[topic] for topic in topics if x[topic] and x[topic].strip()])
        
def parse_single_csv(path:str,slim:bool=True, save:bool=False,verbose:bool=False)->DataFrame:
    '''
    Takes a single CSV in agreed format (as supplied by Gilad) and return a dataframe
    set slim=True to narrow the dataframe to what is needed (fragments & labels)
    set save=True to save a pickle of the dataframe to Data/pickle
    set verbose=True for informative printing
    '''
    if verbose:
        print(f'Parsing {path}')
    df = pd.read_csv(path,encoding='unicode_escape', keep_default_na=False)
    if slim:
        df.rename(columns={'Line':'Fragment'},inplace=True)
        df['Topics'] = df.apply(combine_topics,axis=1)
        df = df[[c for c in df.columns if c in {'Topics','Fragment'}]]  
    if save:
        pkl_path = path.replace(f'csv',f'pickle\\{"Labeled" if is_labeled(df) else "Empty"}')
        os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
        with open(pkl_path, 'wb') as pkl:
            pickle.dump(df,pkl)
            if verbose:
                print(f'\t Saved DataFrame as {pkl.name}')

    return df

def parse_all_csv_in_directory(dir_path:str,slim:bool=True,save:bool=False,verbose:bool=False)->Sequence[tuple[DataFrame,str]]:
    '''
    Parse all CSVs in folder and return a sequence of tuples: (df,filename)
    '''
    return tuple(
        (parse_single_csv(f'{dir_path}{os.sep}{file}',slim,save,verbose),file.split('.')[0])
        for file in os.listdir(dir_path) if file.endswith('csv')
        )

def unpickle_dir(dir_path:str)->Sequence[tuple[DataFrame,str]]:
    '''
    Unpickle all the pickled-dataframes and return a sequence of tuples: (df,filename)
    '''
    return tuple(
        (pickle.load(open(f'{dir_path}{os.sep}{file}','rb')),file.split('.')[0])
        for file in os.listdir(dir_path) if file.endswith('pickle')
    )

def is_labeled(df:DataFrame)->bool:
    '''
    Returns True if there are Topics mentioned for any fragment
    '''
    return any(df['Topics'])

def is_for_filling(df:DataFrame)->bool:
    '''
    Returns True if all of the Topics are blank
    '''
    return not any(df['Topics'])

def unlabeled_topics():
    '''
    Print the topics that had no Label
    '''
    if not MISSING_LABELS:
        print(f'\t > No Missing Labels! Hooray')
    else:
        print(f'\t > Could Not Label: {", ".join([f"`{x}`" for x in MISSING_LABELS])}')


def main()->None:
    dfs,plays = zip(*parse_all_csv_in_directory("data\csv", save=True))
    print(f'Parsed {len(dfs)} CSV: {len([df for df in dfs if is_labeled(df)])} of them are for training:')
    print(f'\t {", ".join([play for df,play in zip(dfs,plays) if is_labeled(df)])}')
    unlabeled_topics()

def get_label_for_word(word: str, allowed_labels: list[Label]) -> Label:
    for label in allowed_labels:
        words_for_label = set([re.sub(r'\W+', '', x.lower()) for x in words_dict[label.name]])
        if word in words_for_label:
           return label
    return Label.NONE

def get_tokens(x: Series)->str:
    labels = x['topics']
    current_index = 1
    words: list[str] = [word.lower() for word in x['text'].split()]
    tokens: list[str] = x['Tags'].split()
    if len(words) != len(tokens):
        print(f"error: {x}")
        return ''
    i = 0
    while i < len(words):
        word = re.sub(r'\W+', '', words[i])
        if tokens[i] == '1':
            also_set_next = False
            if i < len(words) - 1 and words[i+1][0] == '[' and tokens[i+1] == '1':
                word = re.sub(r'\W+', '', words[i+1])
                also_set_next = True
            label = get_label_for_word(word, labels[:current_index])
            if label == Label.NONE:
                tokens[i] = labels[current_index - 1].name
                if also_set_next:
                    also_set_next = False
                    label = get_label_for_word(re.sub(r'\W+', '', words[i]), labels[:current_index])
                    if label != Label.NONE:
                        tokens[i] = label.name
                # print(f"couldn't get label for word: {word}")        
            else:
                tokens[i] = label.name
            if also_set_next:
                tokens[i+1] = tokens[i]
                i += 1
            if current_index < len(labels) and label == labels[current_index - 1]:
                current_index += 1
        i += 1
    return " ".join(tokens)

def parse_csv_token_classification(path: str, fix_spelling=False)->DataFrame:
    '''
    Takes a single CSV in agreed format (as supplied by Gilad) and return a dataframe
    set slim=True to narrow the dataframe to what is needed (fragments & labels)
    set save=True to save a pickle of the dataframe to Data/pickle
    set verbose=True for informative printing
    '''
    df = pd.read_csv(path,encoding='unicode_escape', keep_default_na=False)
    df.rename(columns={'Fragment':'text'},inplace=True)
    if fix_spelling:
        df['text'] = df['text'].apply(rephrase)
    df['topics'] = df.apply(combine_topics,axis=1)
    df['labels'] = df.apply(get_tokens,axis=1)
    df = df[[c for c in df.columns if c in {'text','labels','topics'}]]  
    return df

def intersect_labels(current_labels: list[Label], allowed_labels: set[str]):
    current_set = set([label.name for label in current_labels])
    return len(current_set.intersection(allowed_labels))

def filter_ignored_labels(df: DataFrame, labels_set: set[str]):
    return df[df['topics'].map(lambda x: intersect_labels(x, labels_set)) > 0].reset_index(drop=True)

if __name__ == "__main__":
    main()


