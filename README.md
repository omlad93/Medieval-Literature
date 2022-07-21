# Renaissance - Literature
Topic Modeling Workshop: Renaissance English Literature
Labeling Figurative Language In order to understand relations between topic in the medieval times.

## Kick off
**The Data**  
Inputs: Fragments (a collection of words)  
Outputs: Labels (choosing up to 7 classes from 103 available labels)  
each class has a set of word that it's corresponding to.  
in addition to actual words Gilad added words surrounded by brackets  
`[<t>]` that refer to the title/meaning of the previous word: such as `Elizabeth [Queen]`  

**Labeled Data**  
The Data was collected and manually labeled by Gilad Guttman.  
It is a collection of ~1500 fragments extracted  from 4 plays.  
Each fragment is labeled with up to 7 classes as Excel sheet.  

**Un-Labeled Data**  
There are ~1500 fragments (in addition to the labeled data) extracted from 4 different plays.  
Also, the un-extracted sentences are also available.  

## BaseLine
to be presented a little bit after passover  Medieval-Literature
| #   | Module     | Comments                          | Owner |
| --- | ---------- | --------------------------------- | ----- |
| 1   | Parse Data | from Excel                        | Omri  |
| 2   | Labeling   |                                   | Iris  |
| 3   | Word2Vec   | Initial Step                      | Erez  |
| 4   | Main       | from csv to csv (waiting for 2&3) | Omri  |

### To Use Word2Vec Embeddings
```Python
from gensim.models import Word2Vec, KeyedVectors
word_vectors = KeyedVectors.load('model/word2vec/w2v-plays.wv')
vector = word_vectors['queen'] # numpy vector of a word
```
for more information and methods, such as similarity calculations, check the docs:
https://radimrehurek.com/gensim/models/keyedvectors.html

`w2v-plays.wv` is a model trained on shakespeare's plays I found online. We can also use a pretrained version, that was trained on a part of Google News, with MUCH more data, but of lesser quality for our use case. To use the google news model:
```Python
import gensim.downloader as gensim_api
word_vectors = gensim_api.load('word2vec-google-news-300')
# The rest is the same
```
  
  
## Dependencies
The code is written in `python 3.9.12` or higher
all the necessary python packages can be installed running on command line:  
(make sure to add those packages to the file)  
`pip install -r requirements.txt`

