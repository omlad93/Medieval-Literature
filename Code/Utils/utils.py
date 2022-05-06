
from enum import Enum, auto
import sys
from path import Path
sys.path.insert(0,str(Path(__file__).parent.parent))
import json
import pickle
from Labels.labels import words_dict
from typing import Any


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


def main():
    save_as(words_dict, "Code/Utils/words_dict_str",FileType.JSON, verbose=True)
    save_as(words_dict, "Code/Utils/words_dict_str",FileType.PICKLE, verbose=True)
    

if __name__ == "__main__":
    main()