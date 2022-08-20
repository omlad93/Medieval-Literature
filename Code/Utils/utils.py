
from enum import Enum, auto
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent.parent))
import json
import pickle
# from Labels.labels import words_dict
from typing import Any, Iterable


class FileType(Enum):
    JSON = auto()
    PICKLE = auto()


def save_as(object:Any,filepath:str, filetype:FileType=FileType.PICKLE,strict:bool=False,verbose:bool=False)->None:
    def convert_dict(d:dict,strict:bool)->bool:
        converted = True
        if isinstance(d,dict) and not strict:
            for k,v in d.items():
                if isinstance(v,set):
                    converted = False
                    d[k]=list(v)
        return converted
    
    addition = ''
    if filetype is FileType.JSON:
        try:
            addition = 'After Converting set->list' if convert_dict(object,strict) else ''
            with open(f'{filepath}.json', 'w', encoding='utf8') as jsn:
                json.dump(object, jsn, indent=4, ensure_ascii=False)
                print_conditional(f'Saved {jsn.name}{addition}',verbose)
            return
        except:
            addition = '\tAfter Failing for JSON'
    with open(f'{filepath}.pickle', 'wb') as pkl:
        pickle.dump(object,pkl)
        print_conditional(f'Saved {pkl.name}{addition}',verbose)


def print_conditional(p:Any,verbose:bool):
    if verbose:
        print(p)


def split_on_condition(seq, condition, idx=0):
    '''
    Takes a sq
    '''
    a, b = [], []
    for item in seq:
        if isinstance(item,Iterable):
            (a if condition(item[idx]) else b).append(item)
        else:
            (a if condition(item) else b).append(item)
    return a, b


def normalized_dot_product(v,w):
    def normalize(x):
        norm=np.linalg.norm(x)
        if norm==0:
            norm=np.finfo(x.dtype).eps
        return x/norm
    return np.dot(normalize(v),normalize(w))


def main():
    # save_as(words_dict, "code/utils/words_dict_str",FileType.JSON, verbose=True)
    # save_as(words_dict, "code/utils/words_dict_str",FileType.PICKLE, verbose=True)
    pass
    

if __name__ == "__main__":
    main()