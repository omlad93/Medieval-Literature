import sys
from typing import Sequence
import numpy as np
from pandas import DataFrame
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent.parent))

from Utils.utils import split_on_condition
from Labels.labels import Label, parse_fragment
from parsing.parse_csv import parse_all_csv_in_directory, is_labeled
import datetime


GRANULARITY = 1/10
MIN_T = 5
MAX_T = 10


def label_plays(*,plays:Sequence[tuple[DataFrame,str]], threshold:float, verbose:bool=False)->None:
    for df,name in plays:
        if verbose:
            print(f"\t > Applying labels on {name} using {threshold=}", end='\t')
        
        df['Labels'] = df.apply(
        lambda row: parse_fragment(row.Fragment, threshold),
        axis = 1
        )
        if verbose:
            print(f"Done")

             
def check_labeling(plays:Sequence[tuple[DataFrame,str]], verbose:bool=False)->list[float]:
    ret_list = []
    for df,name in plays:
        if is_labeled(df):
            gold = df['Topics'].to_numpy()
            preds = df['Labels'].to_numpy()
            correct = sum([len(np.intersect1d(gold[i], preds[i])) for i in range(len(gold))])
            total_gold = sum([len(labels) for labels in gold])
            total_preds = sum([len(labels) for labels in preds])
            p = correct / total_preds
            r = correct / total_gold
            f1 = 2 * p * r / (p + r) if correct > 0 else 0
            if verbose:
                print(f'\t > f1 for {name} is {f1:.2f}')
            ret_list.append(f1)
        elif verbose:
            print(f'\t > {name} is not for training, can`t calculate mismatch')
    return ret_list


def main():
    e = datetime.datetime.now()
    print (f'\nStarting  Run {e.strftime("%Y-%m-%d %H:%M:%S")}\n')

    # convert_words_dict_to_vec_dict()
    filled,empty = split_on_condition(parse_all_csv_in_directory("data\csv", save=True),is_labeled,idx=0)
    print(f'Found {len(filled)} Filled plays, and {len(empty)} Empty plays.')
    for t in range(MIN_T,MAX_T):
        threshold=t*GRANULARITY
        label_plays(plays=filled,threshold=threshold,verbose=True)
        performance=check_labeling(filled)
        print (f'{threshold= :.3f}:\t{performance=}')

    e = datetime.datetime.now()
    print (f'\nFinishing Run {e.strftime("%Y-%m-%d %H:%M:%S")}\n')


if __name__ == "__main__":
    main()

