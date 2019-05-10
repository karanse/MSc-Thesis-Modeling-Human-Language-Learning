#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 23:04:29 2019

@author: semakaran
"""

###################### DATA PROCESSING FUNCTIONS ##########################3###

import numpy as np


def make_vocab(data_dict):
    '''
    Input: A nested dictionary (3-level)
    Output: A tuple with 2 elements;
    a vocabulary list and 
    a dictionary of vocabularies with frequencies 
    
    '''
    freq = {}
    for file in data_dict.keys():
        for obj in data_dict[file].keys():
            word = data_dict[file][obj]['word']
            if word in freq:
                freq[word] += 1
            else:
                freq[word] = 1
    vocab = list(freq.keys())
    
    return vocab, freq



def make_ix_table(vocabulary):
    '''
    
    input: a list of vocabulary
    output: a dictionary of vocabularies with the indexes as in the input
    
    '''
    
    word_to_ix = {'<UNK>':0}
    index = 1
    
    for word in vocabulary:
        word_to_ix[word] = index
        index += 1
    
    return word_to_ix


def no_of_objs(data_dict, data_split):
    '''
    Returns a dictionary with the
    number of objects per image
    for a train/validation/test
    split of the data
    '''

    no_of_objs = {}

    for file in data_split:
        if len(data_dict[file]) not in no_of_objs:
            no_of_objs[len(data_dict[file])] = []
            no_of_objs[len(data_dict[file])].append(file)
        else:
            no_of_objs[len(data_dict[file])].append(file)

    return no_of_objs


def get_word_ix(word_to_ix, word):
    '''
    
    input: a dictionary of vocabular with the indexes and a word
    output: an integer as the index of the word in the input
    
    '''
    if word in word_to_ix:
        return word_to_ix[word]
    else:
        return word_to_ix['<UNK>']
    

def dict_to_batches(no_objs_split, bsz):
    '''
    Returns a list of batches. A batch is a
    batch-size lists of file/img ids, of imgs
    containing the same amount of objects.
    The batches are shuffled so that batches
    of different amounts of objects follow
    each other.
    '''
    batch_list = []

    for num in no_objs_split.keys():
        batch_list.extend([no_objs_split[num][x:x+bsz] for x in range(0, len(no_objs_split[num]), bsz)])

    np.random.shuffle(batch_list,)

    return batch_list  
    
    
 ############################################   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    