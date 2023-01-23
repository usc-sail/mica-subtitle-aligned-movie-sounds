'''
Author/year: Rajat Hebbar, 2020

    Use cleaned-up tag->label dictionary to update the 
    labels file for all sound-events.

Input
    movie_sounds (csv-file): sound-events list
    merged_annot (csv-file): annotated events, with category labels
    word_dict (dict): dictionary mapping tags to lemmatized versions 

Output 
    movie_sounds (csv-file): sound-events list, with category labels (sound, source, quality)

Usage
    python add_labels_to_csv.py

'''


import os, sys, numpy as np
import glob
sys.path.insert(0,'/proj/rajatheb/utils/')
from read_files import magic_file_reader as read
import pandas as pd
#os.environ['CUDA_VISIBLE_DEVICES']=''
#import tensorflow as tf


def clean_single_event(aud_event):
    event = aud_event.rstrip().replace('<i>', '').replace('</i>', '').replace('|', 'i').replace('/', 'l').replace('♪', '')
    
    events_fix_paranthesis = [event]    
    if '(' in event and ')' in event:
        if event.index('(') >= event.index(')'):
            new_event1 = event[:event.index(')')]
            new_event2 = event[event.index('(')+1:]
            events_fix_paranthesis = [new_event1, new_event2]

    events_fix_semicolon = []
    for event in events_fix_paranthesis:
        if ';' in event:
            new_events = [x.rstrip().lstrip() for x in event.split(';')]
            events_fix_semicolon.extend(new_events)
        else:
            events_fix_semicolon.append(event)

    return events_fix_semicolon

merged_annot = pd.read_csv('../data/merged.csv')
movie_sounds = pd.read_csv('../data/movie_sounds_2014-18_collar.csv')
word_dict = pd.read_csv('../data/transform.csv')

annotated_events = merged_annot.audio_tag.values

snd = []
src = []
q = []

for sound_event in movie_sounds.sound_event:
#    print(sound_event)
    q_ = []
    src_ = []
    snd_ = []
    for event in clean_single_event(sound_event):
        if event in annotated_events:
            ann_event = merged_annot[annotated_events == event]
            if ann_event['source'].values[0] !=  'None':
                for event_sc in ann_event['source'].values[0].split(';'):
                    src_.append(word_dict[word_dict['word'] == event_sc]['source'].values[0])
            if ann_event['sound'].values[0] !=  'None':
                for event_sc in ann_event['sound'].values[0].split(';'):
                    snd_.append(word_dict[word_dict['word'] == event_sc]['sound'].values[0])
            if ann_event['quality'].values[0] !=  'None':
                for event_sc in ann_event['quality'].values[0].split(';'):
                    q_.append(word_dict[word_dict['word'] == event_sc]['quality'].values[0])
    src.append(src_)    
    snd.append(snd_)
    q.append(q_)

src_str = ["" if x==[] else x[0] if len(x)==1 else ';'.join(x) for x in src]       
snd_str = ["" if x==[] else x[0] if len(x)==1 else ';'.join(x) for x in snd]       
q_str = ["" if x==[] else x[0] if len(x)==1 else ';'.join(x) for x in q]       

movie_sounds['source'] = src_str
movie_sounds['sound'] = snd_str
movie_sounds['quality'] = q_str

movie_sounds.to_csv('../data/movie_sounds_2014-18_tags.csv')


