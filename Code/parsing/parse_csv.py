from __future__ import annotations
import sys
from pathlib import Path
from typing import Sequence
sys.path.insert(0,str(Path(__file__).parent.parent))
import os
import pandas as pd
from pandas import DataFrame
from Labels.labels import Label
import pickle

missing_labels = set()

def convert_labels(topic_list: list[str])->list[Label]:
    '''
    A function to make Labels from Topics (strings)
    '''
    for i,topic in enumerate(topic_list):
        try:
            topic_list[i]=Label[topic.upper()]
        except KeyError as ke:
            missing_labels.add(topic)
    return topic_list
   
def combine_topics(x) -> list[str]:
    '''
    A function to combine all Topics columns into a single column
    While
    '''
    topics = [c for c in x.index if 'Topic' in c]
    return convert_labels([x[topic] for topic in topics if x[topic]])
        
def parse_single_csv(path:str,slim:bool=True, save:bool=False,verbose:bool=False)->DataFrame:
    if verbose:
        print(f'Parsing {path}')
    df = pd.read_csv(path,encoding='unicode_escape', keep_default_na=False)
    if slim:
        df.rename(columns={'Line':'Fragment'},inplace=True)
        df['Topics'] = df.apply(combine_topics,axis=1)
        df = df[[c for c in df.columns if c in {'Topics','Fragment'}]]  
    if save:
        with open(path.replace('csv','pickle'), 'wb') as pkl:
            pickle.dump(df,pkl)
            if verbose:
                print(f'\t Saved DataFrame as {pkl.name}')
    return df

def parse_all_csv_in_directory(dir_path:str,slim=True,save:bool=False,verbose:bool=False)->Sequence[tuple[DataFrame,str]]:
    return tuple(
        (parse_single_csv(f'{dir_path}{os.sep}{file}',slim,save,verbose),file.split('.')[0])
        for file in os.listdir(dir_path) if file.endswith('csv')
        )

def unpickle_dir(dir_path:str)->Sequence[tuple[DataFrame,str]]:
    return tuple(
        (pickle.load(open(f'{dir_path}{os.sep}{file}','rb')),file.split('.')[0])
        for file in os.listdir(dir_path) if file.endswith('pickle')
    )

def is_for_training(df:DataFrame)->bool:
    return any(df['Topics'])

def is_for_filling(df:DataFrame)->bool:
    return not any(df['Topics'])


def main():
    # parse_single_csv("Data//csv//Macbeth.csv")
    # parse_all_csv_in_directory("Data\csv",save=True)
    parse_all_csv_in_directory("Data\csv", save=True)
    print()
    for i,(df,play) in enumerate(unpickle_dir("Data\pickle"),start=1):
        print(f'{i}) {play:<37} is for {"Training" if is_for_training(df) else "Filling"}')

if __name__ == "__main__":
    main()