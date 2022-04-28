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
Also, the un-extracted sentences are also available.  

## BaseLine
to be presented a little bit after passover  Medieval-Literature
| #   | Action     | Comments     | Owner |
| --- | ---------- | ------------ | ----- |
| 1   | Parse Data | from Excel   | Omri  |
| 1   | Labeling   |              | Iris  |
| 2   | Word2Vec   | Initial Step | Erez  |

### TODO:
| Action | Comments | Owner |
| ------ | -------- | ----- |
|        |          | Omri  |
|        |          | Iris  |
|        |          | Erez  |
## Dependencies
The code is written in `python 3.9.12` or higher
all the necessary python packages can be installed running on command line:  
(make sure to add those packages to the file)  
`pip install -r requirements.txt`

