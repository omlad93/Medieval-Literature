from __future__ import annotations
import sys
from pathlib import Path
from typing import Sequence
sys.path.insert(0,str(Path(__file__).parent.parent))
import configargparse as parser
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
        

def parse_single_csv(path:str,slim:bool=True, save:bool=False)->DataFrame:
    print(f'Parsing {path}')
    df = pd.read_csv(path,encoding='unicode_escape', keep_default_na=False)
    if slim:
        df['Topics'] = df.apply(combine_topics,axis=1)
        df = df[[c for c in df.columns if c in {'Topics','Fragment'}]]  
    if save:
        with open(path.replace('csv','pickle'), 'wb') as pkl:
            pickle.dump(df,pkl)
            print(f'\t Saved DataFrame as {pkl.name}')
    return df

def parse_all_csv_in_directory(dir_path:str,slim=True,save:bool=False)->Sequence(DataFrame):
    return tuple(
        parse_single_csv(f'{dir_path}{os.sep}{file}',slim,save) 
        for file in os.listdir(dir_path) if file.endswith('csv')
        )




def main():
    # parse_single_csv("Data//csv//Macbeth.csv")
    # parse_all_csv_in_directory("Data\csv",save=True)
    parse_all_csv_in_directory("Data\csv")
    print('Done')

if __name__ == "__main__":
    main()