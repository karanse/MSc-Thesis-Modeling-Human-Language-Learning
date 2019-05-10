#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 14:23:41 2019

@author: semakaran
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def sum_weighted(objects, attention):
    '''
    input:
    objects: batch X objects in image X visual fetaures
    attention: batch X weights for each object X dummy dim of size 1
    
    returns:
    matrix: batch X weighted sum of visual features for img for each object, 
    the wighting is done on the whole vector of visual features at once.
    If input is softmaxes, therefore, sum of dim 1 should be 1 for each item.
    
    '''
    
    assert len(attention.shape) == 3, "attention should have 3 dimensions: batch * n_objects * 1"
    assert objects.shape[0] == attention.shape[0], "object and attention dim 0 should present batch"
    assert objects.shape[1] == attention.shape[1], "object and attention dim 1 should be nr of objects"
    assert attention.shape[2] == 1, "attention dim 2 should be 1 (dummy dimension)"
    
    weighted = attention * objects
    summed = torch.sum(weighted, dim = 1)
    return summed
    
    

class Speaker(nn.Module):
    def __init__(self, object_size, vocab_size, hidden_size):
        super(Speaker, self).__init__() #check this is necessary
        
        
         # layer that embeds visual weighted summed features into space of correct lstm_size
        self.hidden = nn.Linear(object_size,hidden_size)
        self.word_logits = nn.Linear(hidden_size, vocab_size)
        
        
        ##### Initializing the weights and biases #####
        
        print("Initializing mapping weights...")
        nn.init.kaiming_normal_(self.hidden.weight, mode = 'fan_in',
                                nonlinearity='relu')
        nn.init.kaiming_normal_(self.word_logits.weight,
                                mode= 'fan_in', nonlinearity='relu')
        print('Initializing bias terms to all 0')
        for name,param in self.named_parameters():
            if 'bias' in name:
                print(name)
                nn.init.constant_(param, 0.0)
        
    def forward(self, objects, attention, apply_softmax = True):
        if apply_softmax:
            attention = F.softmax(attention, dim = 1)
            
        # produce weigted sum over visual features
        # to do this you need to expand attention
        attention = attention.unsqueeze(dim = 2)
        weighted_sum = sum_weighted(objects, attention)
        hidden = F.relu(self.hidden(weighted_sum))
        word_probs = self.word_logits(hidden)
        return word_probs
        
    
class Listener(nn.Module):
    def __init__(self, n_objects, object_size, vocab_size, wordemb_size, att_hidden_size):
        
        # Inherit from torch.nn.Module
        super(Listener, self). __init__()
        
        self.word_embedder = nn.Embedding(vocab_size, wordemb_size)
        
        # Producer of attention over visual inputs.
        # MLP with concatenation of lstm output and object as input and 1 output
        self.att_hidden = nn.Linear((wordemb_size + object_size), att_hidden_size)
        self.attention = nn.Linear(att_hidden_size, 1)
        
        
        ##### Initializing the weights and biases #####
        print('Initializing word embeddings')
        nn.init.xavier_uniform_(self.word_embedder.weight, gain = 1)
        
        print('Initializing attention MLP weights...')
        nn.init.kaiming_normal_(self.att_hidden.weight, mode='fan_in', nonlinearity = 'relu')
        nn.init.kaiming_normal_(self.attention.weight, mode = 'fan_in', nonlinearity = 'relu')
        
        print('Initializing bias terms to all 0...')
        for name, param in self.named_parameters():
            if 'bias' in name:
                print(name)
                nn.init.constant_(param, 0.0)
        
    def forward(self, language_input, objects):
        batchsize = language_input.shape[0]
        n_objects = objects.shape[1]
        
        # get language representation
        embeds = self.word_embedder(language_input)
        
        # repeat language n_objects times along the batch dimension
        words = embeds.repeat(n_objects, 1)
        
        # collapse object & batch dimension so that it can go through MLP
        objects = objects.reshape((batchsize*n_objects), objects.shape[2])
        
        # concatanate words and objects
        concat = torch.cat((words, objects), dim = 1)
        
        # return attention  vector
        att_hid = F.relu(self.att_hidden(concat))
        attended = self.attention(att_hid)
        
        # reshape for nlloss & squeeze
        attended = attended.reshape((batchsize, n_objects, 1)).squeeze(2)
        return attended
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        