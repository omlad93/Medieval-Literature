from __future__ import annotations
import sys
from pathlib import Path
from typing import Sequence
sys.path.insert(0,str(Path(__file__).parent.parent))
import os
import pandas as pd # type: ignore
from pandas import DataFrame,Series
from labels.labels import Label, words_dict
from parsing.words_correction import rephrase
import pickle

MISSING_LABELS :set[str] = set()

def convert_labels(topic_list: list[str])->list[Label]:
    '''
    A function to make Labels from labels (strings)
    '''
    for i,topic in enumerate(topic_list):
        label = Label.get_label(topic)
        if label == Label.NONE:
            MISSING_LABELS.add(topic)
        topic_list[i] = label
    return topic_list
   
def combine_topics(x:Series) -> list[str]:
    '''
    A function to combine all labels columns into a single column
    '''
    topics = [c for c in x.index if 'Topic' in c]
    return convert_labels([x[topic] for topic in topics if x[topic] and x[topic].strip()])
        
def parse_single_csv(path:str,slim:bool=True, save:bool=False,verbose:bool=False,fix_spelling:bool=False)->DataFrame:
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
        if "Line" in df.columns:
            df.rename(columns={'Line':'text'},inplace=True)
        if "Fragment" in df.columns:
            df.rename(columns={'Fragment':'text'},inplace=True)
        df['labels'] = df.apply(combine_topics,axis=1)
        df = df[[c for c in df.columns if c in {'labels','text'}]]  
    if save:
        pkl_path = path.replace(f'csv',f'pickle\\{"Labeled" if is_labeled(df) else "Empty"}')
        os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
        with open(pkl_path, 'wb') as pkl:
            pickle.dump(df,pkl)
            if verbose:
                print(f'\t Saved DataFrame as {pkl.name}')
    if fix_spelling:
        df['text'] = df['text'].apply(rephrase)

    return df

def parse_all_csv_in_directory(dir_path:str,slim:bool=True,save:bool=False,verbose:bool=False,fix_spelling:bool=False)->Sequence[tuple[DataFrame,str]]:
    '''
    Parse all CSVs in folder and return a sequence of tuples: (df,filename)
    '''
    return tuple(
        (parse_single_csv(f'{dir_path}{os.sep}{file}',slim,save,verbose,fix_spelling),file.split('.')[0])
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
    Returns True if there are labels mentioned for any fragment
    '''
    return any(df['labels'])

def is_for_filling(df:DataFrame)->bool:
    '''
    Returns True if all of the labels are blank
    '''
    return not any(df['labels'])

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

if __name__ == "__main__":
    main()


