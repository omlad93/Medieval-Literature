import sys
from typing import Sequence
import numpy as np
from pandas import DataFrame
from path import Path
sys.path.insert(0,str(Path(__file__).parent.parent))

from utils.utils import split_on_condition
from labels.labels import Label, parse_fragment # TODO: verify parse_fragment
from parsing.parse_csv import parse_all_csv_in_directory, is_labeled, convert_labels



def loss(original:Sequence[Label],applied:Sequence[Label], miss_w=1.0,extra_w=1.0):
    '''
    compare two label-sets of a fragment. 
    given 2 labels sequence (1 hand labeled and 1 by this  scripts):
    count missing labels (labeled by hand and not by script) and added labels (labeled by script and not by hand)
    sum them up with matching weights
    '''
    misses = [item for item in original if item not in applied]
    extras = [item for item in applied if item not in original]
    return miss_w*len(misses) + extra_w*len(extras)


def label_plays(plays:Sequence[tuple[DataFrame,str]]):
    ## TODO FIXME: use actual labeling function according to vectors.
    ## Instead of `parse_fragment()`
    for df,name in plays:
        # Label it again
        df['Labels'] = df.apply(
        lambda row: convert_labels(parse_fragment(row.Fragment)),
        axis = 1
        )
        print(f"\t > Applied labels on {name}")

            
def check_labeling(plays:Sequence[tuple[DataFrame,str]]):
    ## TODO FIXME: use actual loss function.
    ## Instead of `loss()`
    for df,name in plays:

        if is_labeled(df):
            df['Loss'] = df.apply(
            lambda row: loss(row.Topics,row.Labels),
            axis = 1
            )
            print(f'\t > Average Diff for {name} is {np.average(df.Loss):.2f}')
        else:
            print(f'\t > {name} is not for training, can`t calculate loss')




def main():
    filled,empty = split_on_condition(parse_all_csv_in_directory("data\csv", save=True),is_labeled,idx=0)
    print(f'Found {len(filled)} Filled plays, and {len(empty)} Empty plays.')
    label_plays(filled)
    check_labeling(filled)

    






if __name__ == "__main__":
    main()

