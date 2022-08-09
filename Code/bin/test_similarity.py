#!/usr/bin/env python3
import fractions
import imp
import sys
from typing import Optional, Sequence
import numpy as np
from pandas import DataFrame
from parso import parse
from path import Path
sys.path.insert(0,str(Path(__file__).parent.parent))

from utils.utils import split_on_condition
from labels.labels import *
from parsing.parse_csv import parse_all_csv_in_directory, is_labeled, convert_labels
from bin.main import check_labeling,label_plays


TH_COUNT= 20
TH_GRANULARITY = 1/TH_COUNT


def get_labels_test(vec:np.array, thrashold:int):
    '''
    calculates the iiner produt with the words in the vectors dictionary and
    returns a set of labels that the inner product were higher than the threshold
    '''
    labels,similarities = [],[]
    for label in vec_dict.keys():
        for tpl in vec_dict[label]:
            try:
                similarity = word_vectors.similarity(tpl,vec)
                #inner_prod = calc_inner_product(tpl[0],vec)
                if similarity > thrashold:
                    labels.append(tuple((label, similarity)))
                    break
            except:
                pass
    return labels

def parse_fragment_test(fragment:str, threshold:float=0.5 ,verbose:bool=False) -> Sequence[Label]:
    '''
    get all labels that the word the the fragment associated with
    set verbose=True for deatiled information of words in fragment and their labels
    '''
    #words = fragment.split().replace(',','').replace('.','').replace('?','').replace('!','').replace(':','').replace(';','')
    labels, similarities = [], []

    temp = fragment.split()
    words = [word.replace(',','').replace('.','').replace('?','').replace('!','').replace(':','').replace(';','') for word in temp]
    for word in words:
        for label,similarity in get_labels_test(word, threshold):
            labels.append(label)
            similarities.append(similarity)
    # labels_list = list(labels)
    # labels_list.sort(key=lambda x:x[1]) #sorting the list by the similarity
    return (labels,similarities)


def normalize(v):
    norm=np.linalg.norm(v)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm

def normalized_dot(v,w):
    return np.dot(normalize(v),normalize(w))

def str_try(couples):
    print('Running str Similarity')
    values = []
    for a,b in couples:
        val = word_vectors.similarity(a , b)
        pre = f"Similarity({a},{b})"
        # print(f"{pre:<30}: {val}")
        values.append(val)
    return sorted(couples,key=lambda t:word_vectors.similarity(t[0] , t[1])),values



def vec_try(couples):
    values =[]
    print('Running vec Similarity')
    for a,b in couples:
        pre = f'Dot(Vec({a}),Vec({b}))'
        val = normalized_dot(word_vectors[a]  , word_vectors[b].T)
        # print(f"{pre:<30}: {val}")
        values.append(val)
    return sorted(couples,key=lambda t:normalized_dot(word_vectors[t[0]]  , word_vectors[t[1]].T)),values



def make_similarity_great_again():

    couples = [
    ('woman'  ,'man'),
    ('woman'  ,'men'),
    ('woman'  ,'apple'),
    ('queen'  ,'king'),
    ('queens' ,'king'),
    ('prince' ,'murder'),
    ('play'   ,'game'),
    ('mouse'  ,'play'),
    ('money'  ,'fruit'),
    ]
    sc, sv=str_try(couples)
    vc, vv=vec_try(couples)
    for i in range(len(couples)):
        print(f'{i:<2}: D={abs(sv[i]-vv[i]):.2f} sc={str(sc[i]):<25}, vc={str(vc[i]):<25}')
    
    

def main():
    convert_words_dict_to_vec_dict()

    fragments = [
        'The king is dead',
        'planted some seeds in the forest',
        'the fish are well for feeding sharks',
        'we run and score. we are winning',
        'to be or not to be, ask stack-overflow since I have no idea'
    ]

    for f in fragments:
        print(f' > Labeling `{f}`')
        for t in range(int(2*TH_COUNT/3),TH_COUNT):
            print(f'\t > Using Threshold {t*TH_GRANULARITY:.2f}:')
            print(f'\t\t{parse_fragment(f,threshold=t*TH_GRANULARITY)}')




if __name__ == "__main__":
    main()

