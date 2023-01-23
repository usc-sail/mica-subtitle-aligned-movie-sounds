from nltk.corpus.reader.wordnet import NOUN
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
from collections import Counter
import pandas as pd
from itertools import chain

csv_file = "../data/merged.csv"
cln = pd.read_csv(csv_file).fillna("None")
cln_sounds = sorted(list(set(cln.sound)))
cln_sources = sorted(list(set(cln.source)))
cln_q = sorted(list(set(cln.quality)))


## Split words by ";" indicating multiple occuring snd/src/qual
sounds = list(chain.from_iterable(x.split(';') if ';' in x else [x] for x in cln_sounds))
sources = list(chain.from_iterable(x.split(';') if ';' in x else [x] for x in cln_sources))
qualities = list(chain.from_iterable(x.split(';') if ';' in x else [x] for x in cln_q))


## Dictionary to map category words to lemmatized form
transform = {'sound': {}, 'source': {}, 'quality': {}}
transform['sound'] = { x: wnl.lemmatize(wnl.lemmatize(x, 'v'), 'n') for x in sorted(sounds) if str(x) != 'None'}
transform['source'] = { x: wnl.lemmatize(x, 'n') for x in sources if str(x) != 'None'}
transform['quality'] = { x: wnl.lemmatize(x, 'n') for x in qualities if str(x) != 'None'}
tf_ = pd.DataFrame(transform)
tf_.index.names = ['word']
tf_ = tf_.reset_index()
tf_.to_csv("../data/transform.csv", index=False)

## Set of all possible category tags
cat_data = {k: sorted(list(set(val.values()))) for k, val in transform.items() }

## Dictionary mapping tags to categories
word_dict = {}
for cat in ['sound', 'source', 'quality']:
    for word in cat_data[cat]:
        if word not in word_dict:
            word_dict[word] = [cat]
        elif cat not in word_dict[word]:
            word_dict[word].append(cat)


## Append cleaned up words to 
