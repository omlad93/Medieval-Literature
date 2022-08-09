import sys
from typing import Optional, Sequence
import numpy as np
from pandas import DataFrame
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent.parent))

from Utils.utils import split_on_condition
from Labels.labels import Label, parse_fragment,convert_words_dict_to_vec_dict # TODO: verify parse_fragment
from parsing.parse_csv import parse_all_csv_in_directory, is_labeled, convert_labels
import datetime




GRANULARITY = 1/10
MIN_T = 5
MAX_T = 10


def mismatch(original:Sequence[Label],applied:Sequence[Label], miss_w=1.0,extra_w=1.0):
    '''
    compare two label-sets of a fragment. 
    given 2 labels sequence (1 hand labeled and 1 by this  scripts):
    count missing labels (labeled by hand and not by script) and added labels (labeled by script and not by hand)
    sum them up with matching weights
    '''
    misses = [item for item in original if item not in applied]
    extras = [item for item in applied if item not in original]
    return miss_w*len(misses) + extra_w*len(extras)


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
    ## TODO FIXME: use actual performance function.
    ## Instead of `loss()`
    ret_list = []
    for df,name in plays:
        if is_labeled(df):
            df['Loss'] = df.apply(
            lambda row: mismatch(row.Topics,row.Labels),
            axis = 1
            )
            if verbose:
                print(f'\t > Average Diff for {name} is {np.average(df.Loss):.2f}')
            ret_list.append(np.average(df.Loss))
        elif verbose:
            print(f'\t > {name} is not for training, can`t calculate mismatch')
    return ret_list




def main():
    e = datetime.datetime.now()
    print (f'\nStarting  Run {e.strftime("%Y-%m-%d %H:%M:%S")}\n')

    convert_words_dict_to_vec_dict()
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

