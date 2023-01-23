'''
Author/year: Rajat Hebbar, 2020

    Create unique sound-event tags list for annotation
    based on frequency of occurrence + clean-up

Input
    csv_file - Labels file

Output 
    dictionary - event tags sorted by frequency

Usage
    python clean_up_event_tags.py

'''


import os, sys, numpy as np, pandas as pd
import glob
sys.path.insert(0,'/data/rajatheb/utils/')
from read_files import magic_file_reader as read
import argparse
from collections import Counter
#os.environ['CUDA_VISIBLE_DEVICES']=''
#import tensorflow as tf

data_file = '/data/movies/movie_sounds_2014_2018_RH_collar_KS.csv'
data = pd.read_csv(data_file)

events_orig = [x.rstrip() for x in data.sound]
events_fix_paranthesis = []
events_fix_semicolon = []


## Remove/replace non alphanumeric characters (not all)
events_orig = [x.replace('<i>', '').replace('</i>', '').replace('|', 'i').replace('/', 'l').replace('♪', '') for x in events_orig]


## A subtitle may contain two events,, e.g: "swoosh) - (eating"
for event in events_orig:
    if '(' in event and ')' in event:
        if event.index('(') >= event.index(')'):
            new_event1 = event[:event.index(')')]
            new_event2 = event[event.index('(')+1:]
            events_fix_paranthesis.extend([new_event1, new_event2])
    else:
            events_fix_paranthesis.append(event)
## A subtitle may contained two events split by ;. e.g: "grunt; groan"
for event in events_fix_paranthesis:
    if ';' in event:
        new_events = [x.rstrip().lstrip() for x in event.split(';')]
        events_fix_semicolon.extend(new_events)
    else:
        events_fix_semicolon.append(event)

uniq_events = Counter(events_fix_semicolon)
uniq_events_sorted = uniq_events.most_common()
unigrams = [ x for x in uniq_events_sorted if x[0].isalnum() ]
bigrams = [ x for x in uniq_events_sorted if len(x[0].split())==2 ]

events = {}

for k in [2,3,4,5,10,20]:
    atleast_k = [x for x in uniq_events_sorted if x[1] >= k]
    events[str(k)] = atleast_k
    num_uniq = len(atleast_k)
    cover = sum([x[1] for x in atleast_k])
    print("{} : {}, {}".format(k, num_uniq, cover))




