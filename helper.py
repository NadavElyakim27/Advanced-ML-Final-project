#------------------------------------IMPORTS AND INSTALLATIONS----------------------------------------#
import re
import os
import sys
import nltk
import spacy
import math
import socket
import torch
import datetime
import logging
import logging.config
import yaml
import string
import warnings
import numpy as np
import preprocessor as twitter_preprocessor
from copy import deepcopy
from nltk.corpus import stopwords
from spacy.symbols import PUNCT, SYM, ADJ, CCONJ, NUM, DET, ADV, ADP, VERB, NOUN, PROPN, PART, PRON, ORTH
from collections import Iterable
from nltk.corpus import stopwords

warnings.filterwarnings("ignore", category=UserWarning, module='bs4')
map_stance_label_to_s = { 0: "support",1: "comment", 2: "deny", 3: "query"}
map_s_to_label_stance = {y: x for x, y in map_stance_label_to_s.items()}


#------------------------------------CREATE BRACNCHES FROM TREE---------------------------------------#
def tree2branches(root):
    
    node = root
    parent_tracker = []
    parent_tracker.append(root)
    branch = []
    branches = []
    i = 0
    siblings = None
    while True:
        node_name = list(node.keys())[i]
        branch.append(node_name)
        first_child = list(node.values())[i]
        if first_child != []:  
            node = first_child  
            parent_tracker.append(node)
            siblings = list(first_child.keys())
            i = 0  
        else:
            branches.append(deepcopy(branch))
            if siblings is not None:
                i = siblings.index(node_name) 
                while i + 1 >= len(siblings):
                    if node is parent_tracker[0]:  
                        return branches
                    del parent_tracker[-1]
                    del branch[-1]
                    node = parent_tracker[-1] 
                    node_name = branch[-1]
                    siblings = list(node.keys())
                    i = siblings.index(node_name)
                i = i + 1  
                del branch[-1]
            else:
                return branches

#------------------------------------PREPROCESS TEXT----------------------------------------#
nlp = None
punctuation = list(string.punctuation) + ["``"]
stopWords = set(stopwords.words('english'))

def preprocess_text(text, opts, nlpengine=None, lang='en_core_web_sm', special_tags=["<pad>", "<eos>"], use_tw_preprocessor=True):
   

    if use_tw_preprocessor:
        text = re.sub(r"(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?", "$URL$",text.strip())
        twitter_preprocessor.set_options('mentions')
        text = twitter_preprocessor.tokenize(text)
        
    if nlpengine is None:
        global nlp
        if nlp is None:
            nlp = spacy.load(lang)
            nlp.add_pipe('sentencizer')
            for x in ['URL', 'MENTION', 'HASHTAG', 'RESERVED', 'EMOJI', 'SMILEY', 'NUMBER', ]:
                nlp.tokenizer.add_special_case(f'${x}$', [{ORTH: f'${x}$'}])
        nlpengine = nlp
        
    processed_chunk = ""
    doc = nlpengine(text)
    doclen = 0
    
    for sentence in doc.sents:
        for w in sentence:
            word = "_".join(w.text.split())
            if word.isspace() or word == "":
                continue
            if opts.remove_stop_words and word.lower() in stopWords:
                continue
            if opts.remove_puncuation and word in punctuation:
                continue
            if opts.lemmatize_words:
                output = w.lemma_ if w.lemma_ != '-PRON-' else w.lower_
            else:
                output = word
            if opts.to_lowercase:
                output = output.lower()
            if opts.replace_nums and output.replace('.', '', 1).isdigit():
                output = opts.num_replacement
            output = output.replace("n't", "not")
            doclen += 1
            processed_chunk += "%s " % (output)

        if opts.add_eos:
            doclen += 1
            processed_chunk += opts.eos + "\n"
        else:
            processed_chunk += "\n"

    processed_chunk = processed_chunk.strip()
    
    return processed_chunk


#------------------------------------TRANSFORM FEATURE TO DICT----------------------------------------#
def transform_feature_dict(thread_feature_dict, conversation, feature_set):
    

    thread_features_array = []
    thread_features_dict = []
    thread_stance_labels = []
    clean_branches = []

    branches = conversation['branches']

    for branch in branches:
        branch_rep = []
        branch_rep_dicts = []
        # contains ids for tweets
        clb = []
        branch_stance_lab = []
        for twid in branch:
            if twid in thread_feature_dict.keys():
                tweet_rep, tweet_rep_dict = dict_to_array_and_dict(thread_feature_dict[twid], feature_set)
                branch_rep.append(tweet_rep)
                branch_rep_dicts.append(tweet_rep_dict)

                # if it is source tweet
                if twid == branch[0]:
                    # if it is labelled
                    if 'label' in list(conversation['source'].keys()):
                        branch_stance_lab.append(convert_label(
                            conversation['source']['label']))
                    clb.append(twid)
                else:
                    for r in conversation['replies']:
                        if r['id_str'] == twid:
                            if 'label' in list(r.keys()):
                                branch_stance_lab.append(
                                    convert_label(r['label']))
                            clb.append(twid)
        if branch_rep != []:
            branch_rep = np.asarray(branch_rep)
            branch_stance_lab = np.asarray(branch_stance_lab)
            thread_features_array.append(branch_rep)
            thread_features_dict.append(branch_rep_dicts)
            thread_stance_labels.append(branch_stance_lab)
            clean_branches.append(clb)

    return thread_features_array, thread_features_dict, thread_stance_labels, clean_branches


#------------------------------------TRANSFORM DICT TO ARRAY----------------------------------------#
def dict_to_array(feature_dict, feature_set):

    tweet_rep = []
    for feature_name in feature_set:

        if np.isscalar(feature_dict[feature_name]):
            tweet_rep.append(feature_dict[feature_name])
        else:
            tweet_rep.extend(feature_dict[feature_name])
    tweet_rep = np.asarray(tweet_rep)
    return tweet_rep


#------------------------------------TRANSFORM DICT TO ARRAY AND DICT----------------------------------------#
def dict_to_array_and_dict(feature_dict, feature_set):

    tweet_rep = []
    tweet_rep_d = dict()
    for feature_name in feature_set:
        tweet_rep_d[feature_name] = feature_dict[feature_name]
        if np.isscalar(feature_dict[feature_name]):
            tweet_rep.append(feature_dict[feature_name])
        else:
            tweet_rep.extend(feature_dict[feature_name])

    tweet_rep = np.asarray(tweet_rep)
    return tweet_rep, tweet_rep_d


#------------------------------------EXSTRACT THREAD FEATURES----------------------------------------#
def extract_thread_features(conversation):
    
    feature_dict = {}
    tw = conversation['source']
    tokens = nltk.word_tokenize(re.sub(r'([^\s\w]|_)+', '', tw['text'].lower()))
    otherthreadtweets = ''
    for response in conversation['replies']:
        otherthreadtweets += ' ' + response['text']
    otherthreadtokens = nltk.word_tokenize(re.sub(r'([^\s\w]|_)+', '', otherthreadtweets.lower()))
    raw_txt = tw['text']
    feature_dict['raw_text'] = raw_txt
    feature_dict['spacy_processed_text']= preprocess_text(raw_txt, initopts())
                              
    return feature_dict

#------------------------------------EXSTRACT THREAD FEATURES AND HIS REPLIES ----------------------------------------#
def extract_thread_features_incl_response(conversation):

    source_features = extract_thread_features(conversation)
    source_features['issource'] = 1
    srctokens = nltk.word_tokenize(re.sub(r'([^\s\w]|_)+', '', conversation['source']['text'].lower()))
    fullthread_featdict = {}
    fullthread_featdict[conversation['source']['id_str']] = source_features

    for tw in conversation['replies']:
        feature_dict = {}
        feature_dict['issource'] = 0
        tokens = nltk.word_tokenize(re.sub(r'([^\s\w]|_)+', '',tw['text'].lower()))
        otherthreadtweets = ''
        otherthreadtweets += conversation['source']['text']

        for response in conversation['replies']:
            otherthreadtweets += ' ' + response['text']

        otherthreadtokens = nltk.word_tokenize(re.sub( r'([^\s\w]|_)+', '',otherthreadtweets.lower()))
        branches = tree2branches(conversation['structure'])

        for branch in branches:
            if tw['id_str'] in branch:
                if branch.index(tw['id_str']) - 1 == 0:
                    prevtokens = srctokens
                else:
                    prev_id = branch[branch.index(tw['id_str']) - 1]
                    # Find conversation text for the id
                    for ptw in conversation['replies']:
                        if ptw['id_str'] == prev_id:
                            prevtokens = nltk.word_tokenize(re.sub(r'([^\s\w]|_)+','', ptw['text'].lower()))
                            break
            else:
                prevtokens = []
            break

        raw_txt = tw['text']
        feature_dict['raw_text'] = raw_txt
        feature_dict['spacy_processed_text'] = preprocess_text(raw_txt, initopts())
        feature_dict['src_usr_hasurl'] = 0
        fullthread_featdict[tw['id_str']] = feature_dict
        
    return fullthread_featdict

            
#------------------------------------GET CLASS WEIGHTS---------------------------------------#
def get_class_weights(examples: Iterable, label_field_name: str, classes: int) -> torch.FloatTensor:

    arr = torch.zeros(classes)
    for e in examples:
        arr[int(getattr(e, label_field_name))] += 1
    arrmax = arr.max().expand(classes)
    
    return arrmax / arr

#------------------------------------HELPER FUNCTIONS---------------------------------------#
def initopts():
    
    o = DotDict()
    o.stopwords_file = ""
    o.remove_puncuation = False
    o.remove_stop_words = False
    o.lemmatize_words = False
    o.num_replacement = "[NUM]"
    o.to_lowercase = False
    o.replace_nums = False  # Nums are important, since rumour may be lying about count
    o.eos = "[EOS]"
    o.add_eos = True
    o.returnNERvector = True
    o.returnDEPvector = True
    o.returnbiglettervector = True
    o.returnposvector = True
    
    return o

#------------------------------------#
class DotDict(dict):

    def __getattr__(self, key):
        return self[key]
    
    def __setattr__(self, key, val):
        if key in self.__dict__:
            self.__dict__[key] = val
        else:
            self[key] = val
            
    def __getstate__(self):
        pass
    
    def __setstate__(self, state):
        pass

#------------------------------------#
def totext(batch, vocab, batch_first=True, remove_specials=False, check_for_zero_vectors=True):
    
    textlist = []
    if not batch_first:
        batch = batch.transpose(0, 1)
    for ex in batch:
        if remove_specials:
            textlist.append(
                ' '.join(
                    [vocab.itos[ix.item()] for ix in ex
                     if ix != vocab.stoi["<pad>"] and ix != vocab.stoi["<eos>"]]))
        else:
            if check_for_zero_vectors:
                text = []
                for ix in ex:
                    if ix != vocab.stoi["<pad>"] and ix != vocab.stoi["<eos>"] \
                            and vocab.vectors[ix.item()].equal(vocab.vectors[vocab.stoi["<unk>"]]):
                        text.append(f"[OOV]{vocab.itos[ix.item()]}")
                    else:
                        text.append(vocab.itos[ix.item()])
                textlist.append(' '.join(text))
            else:
                textlist.append(' '.join([vocab.itos[ix.item()] for ix in ex]))
                
    return textlist

#------------------------------------#
def count_parameters(model):
    
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#------------------------------------#
def convert_label(label):
    if label == "support":
        return (0)
    elif label == "comment":
        return (1)
    elif label == "deny":
        return (2)
    elif label == "query":
        return (3)
    else:
        print(label)

# global model_GN
# model_GN = None

