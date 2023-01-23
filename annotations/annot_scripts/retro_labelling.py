## Check all possible sub-string matches b/w existing tags and new events
def common_seq_words(str1, str2):
    words_str1 = str1.split(" ")
    l = len(words_str1) + 1
    for i in range(l):
        for j in range(i+1, l):
            if " ".join(words_str1[i:j]) == str2:
                return True
    return False


## Label events_10 (b/w 10 and 20 occ) using annotations from events_20
events_10 = list(set(events['10']) - set(events('20')))
word_dict_uniq = {k: v[0] for k, v in word_dict.items()}
events_10_cats = {k:{} for k in events_10}
for event in events_10:
    for word, cat in word_dict_uniq.items():
        noun_form = " ".join([wnl.lemmatize(x, 'n') for x in event.split(" ")])
        verb_form = " ".join([wnl.lemmatize(x, 'v') for x in event.split(" ")])
        if common_seq_words(event, word) or common_seq_words(noun_form, word) or common_seq_words(verb_form, word):
            if cat in events_10_cats[event]:
                events_10_cats[event][cat].append(word)
            else:
                events_10_cats[event][cat] = [word]
