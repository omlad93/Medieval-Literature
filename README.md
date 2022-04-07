# Medieval-Literature
Topic Modeling Workshop: Medieval English Literature
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
Also, the unextracted senteces are also available.  

## BaseLine
to be presented a little bit after passover  
| #   | action     | Comments     | Owner |
| --- | ---------- | ------------ | ----- |
| 1   | parse data | and Labels   | TBD   |
| 2   | Word2Vec   | Initial Step | TBD   |

