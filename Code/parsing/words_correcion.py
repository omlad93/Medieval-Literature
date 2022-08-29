from __future__ import annotations
#from curses.ascii import NUL
from enum import Enum,auto
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent.parent))
from typing import Any, Optional, Sequence
from difflib import get_close_matches
import re
import numpy as np
# from gensim.models import Word2Vec, KeyedVectors
from Utils.utils import normalized_dot_product

correct_words_dict : dict[str,str] = {
    "lefte" : "left",
    "eies" : "eyes",
    "loue" : "love",
    "soule" : "soul",
    "saluation" : "solution",
    "liue" : "live",
    "aliue" : "alive",
    "euery" : "every",
    "soull" : "soul",
    "sicke" : "sick",
    "inuest" : "invest",
    "doe" : "do",
    "harte" : "hart",
    "giuen vp" : "given up",
    "giuen" : "given",
    "vp": "up",
    "voyce" : "voice",
    "goe" : "go",
    "musick" : "music",
    "walke" : "walk",
    "haue" : "have",
    "euer" : "ever",
    "backe" : "back",
    "ouer" : "over", 
    "turne" : "turn",
    "liue for euer" : "live forever",
    "milke" : "milk",
    "necke" : "neck",
    "losse" : "loss",
    "sleepe" : "sleep",
    "iuice" : "juice",
    "clocke" : "clock",
    "finde" : "find",
    "sicke" : "sick",
    "wind" : "wind", 
    "leaue" : "leave",
    "heauen" : "heaven",
    "lesse" : "less",
    "bloud" : "blood"  
}


def word_correction(src_word):
    if src_word in correct_words_dict.keys(): return correct_words_dict[src_word]
    return src_word