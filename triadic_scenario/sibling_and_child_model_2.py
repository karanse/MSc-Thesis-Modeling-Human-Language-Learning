import random
random.seed(a=456)
# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
# setting torch seeds
torch.cuda.manual_seed(456)
torch.manual_seed(456)
# general imports
import json
import pickle
import time
#import csv #?
import numpy as np
np.random.seed(456)
import sys
# custom imports
from helper import make_vocab, make_ix_table, no_of_objs, get_word_ix, dict_to_batches
import child_agent



################## EVALUATION FUNCTION ########################################

def evaluate(epoch, split='val'):
    listener_child.eval()
    speaker_child.eval()
#    listener_sibling.eval()
    speaker_sibling.eval()
    
    if split=='val':
        batchlist = val_batchlist
    elif split=='test':
        batchlist = test_batchlist
    n_batches = len(batchlist)
    start_time = time.time()
    
    # create empty arrays for SIBLING eval
#    li_eval_sibling_loss = np.empty(n_batches)
#    li_eval_sibling_acc = np.empty(n_batches)
    sp_eval_sibling_loss = np.empty(n_batches)
    sp_eval_sibling_acc = np.empty(n_batches)
    
    # create empty arrays for CHILD eval
    li_eval_child_loss = np.empty(n_batches)
    li_eval_child_acc = np.empty(n_batches)
    sp_eval_child_loss = np.empty(n_batches)
    sp_eval_child_acc = np.empty(n_batches)
    
    batch_size = np.empty(n_batches)

    batch = 0

    while batch < n_batches:
        language_input, visual_input, targets = load_val_batch(dict_words_boxes,
                                                        batchlist[batch],
                                                        word_to_ix,
                                                        device)
########## Sibling evaluation ##################################################
        
#        obj_guesses_sibling = listener_sibling(language_input, visual_input) # obj_guesses van maken
#        obj_guess_values_sibling = obj_guesses_sibling.detach()
        
#        obj_guess_values_sibling = torch.nn.functional.one_hot(language_input)
        
  
        obj_guess_values_sibling = one_hot_embedding(targets, visual_input.shape[1]).to(device)

        word_guesses_sibling = speaker_sibling(visual_input, obj_guess_values_sibling)
        
        sibling_language_output = torch.argmax(word_guesses_sibling ,dim =1)
        
#        li_sibling_loss = criterion(obj_guesses_sibling, targets)
#        li_eval_sibling_acc[batch], batch_size[batch] = calc_accuracy(obj_guesses_sibling, targets)
#        li_eval_sibling_loss[batch] = li_sibling_loss.item() * batch_size[batch]
#        li_eval_sibling_acc[batch] *= batch_size[batch] #avg weighted for differing batchsizes

        sp_sibling_loss = criterion(word_guesses_sibling, language_input)
        sp_eval_sibling_loss[batch] = sp_sibling_loss.item() * batch_size[batch]
        sp_eval_sibling_acc[batch],_ = calc_accuracy(word_guesses_sibling, language_input)
        sp_eval_sibling_acc[batch] *= batch_size[batch] #avg weighted for differing batchsizes
        
############### Child evaluation ##############################################
        
        language_input_child = targets_1_1(language_input, sibling_language_output).to(device)
        
                
        obj_guesses_child = listener_child(language_input_child, visual_input) # obj_guesses van maken
        obj_guess_values_child = obj_guesses_child.detach()

        word_guesses_child = speaker_child(visual_input, obj_guess_values_child)
        
        
        li_child_loss = criterion(obj_guesses_child, targets)
        li_eval_child_acc[batch], batch_size[batch] = calc_accuracy(obj_guesses_child, targets)
        li_eval_child_loss[batch] = li_child_loss.item() * batch_size[batch]
        li_eval_child_acc[batch] *= batch_size[batch] #avg weighted for differing batchsizes

        sp_child_loss = criterion(word_guesses_child, language_input)
        sp_eval_child_loss[batch] = sp_child_loss.item() * batch_size[batch]
        sp_eval_child_acc[batch],_ = calc_accuracy(word_guesses_child, language_input)
        sp_eval_child_acc[batch] *= batch_size[batch] #avg weighted for differing batchsizes
        
        batch += 1
        if batch % printerval == 0:
            print('| epoch {:2d} | batch {:3d}/{:3d} | t {:6.2f} | s.S.L {:6.4f} | s.S.A {:5.4f} | l.C.L {:6.4f} | l.C.A {:5.4f} | s.C.L {:6.4f} | s.C.A {:5.4f} |'.format(
                epoch, batch, n_batches, (time.time() - start_time),
#                np.sum(li_eval_sibling_loss[batch-printerval:batch])/np.sum(batch_size[batch-printerval:batch]),
#                np.sum(li_eval_sibling_acc[batch-printerval:batch])/np.sum(batch_size[batch-printerval:batch]),
                np.sum(sp_eval_sibling_loss[batch-printerval:batch])/np.sum(batch_size[batch-printerval:batch]),
                np.sum(sp_eval_sibling_acc[batch-printerval:batch])/np.sum(batch_size[batch-printerval:batch])),
            
                np.sum(li_eval_child_loss[batch-printerval:batch])/np.sum(batch_size[batch-printerval:batch]),
                np.sum(li_eval_child_acc[batch-printerval:batch])/np.sum(batch_size[batch-printerval:batch]),
                np.sum(sp_eval_child_loss[batch-printerval:batch])/np.sum(batch_size[batch-printerval:batch]),
                np.sum(sp_eval_child_acc[batch-printerval:batch])/np.sum(batch_size[batch-printerval:batch]))
#             
#    avg_li_eval_sibling_loss = np.sum(li_eval_sibling_loss) / np.sum(batch_size)
#    avg_li_eval_sibling_acc = np.sum(li_eval_sibling_acc) / np.sum(batch_size)
    avg_sp_eval_sibling_loss = np.sum(sp_eval_sibling_loss) / np.sum(batch_size)
    avg_sp_eval_sibling_acc = np.sum(sp_eval_sibling_acc) / np.sum(batch_size)
    
    avg_li_eval_child_loss = np.sum(li_eval_child_loss) / np.sum(batch_size)
    avg_li_eval_child_acc = np.sum(li_eval_child_acc) / np.sum(batch_size)
    avg_sp_eval_child_loss = np.sum(sp_eval_child_loss) / np.sum(batch_size)
    avg_sp_eval_child_acc = np.sum(sp_eval_child_acc) / np.sum(batch_size)

    if split == 'val':
        print('-' * 89)
        print("overall performance on validation set:")
        print('| L.C.loss {:8.4f} | L.C.acc. {:8.4f} |'.format(
#            avg_li_eval_sibling_loss,
#            avg_li_eval_sibling_acc,
            avg_li_eval_child_loss,
            avg_li_eval_child_acc))
        print('| S.S.loss {:8.4f} | S.S.acc. {:8.4f} | S.C.loss {:8.4f} | S.C.acc. {:8.4f} |'.format(
            avg_sp_eval_sibling_loss,
            avg_sp_eval_sibling_acc,
            avg_sp_eval_child_loss,
            avg_sp_eval_child_acc))
        print('-' * 89)
    elif split == 'test':        
        print('-' * 89)
        print("overall performance on test set:")
        print('| L.C.loss {:8.4f} | L.C.acc. {:8.4f} |'.format(
#            avg_li_eval_sibling_loss,
#            avg_li_eval_sibling_acc,
            avg_li_eval_child_loss,
            avg_li_eval_child_acc))
        print('| S.S.loss {:8.4f} | S.S.acc. {:8.4f} | S.C.loss {:8.4f} | S.C.acc. {:8.4f} |'.format(
            avg_sp_eval_sibling_loss,
            avg_sp_eval_sibling_acc,
            avg_sp_eval_child_loss,
            avg_sp_eval_child_acc))
        print('-' * 89)
    return avg_li_eval_child_loss, avg_li_eval_child_acc, avg_sp_eval_sibling_loss, avg_sp_eval_sibling_acc, avg_sp_eval_child_loss, avg_sp_eval_child_acc



################## FUNCTIONS FOR DATA LOADING #################################
    


def load_val_batch(dict_words_boxes, batch, word_to_ix, device):
    # Loads the batches for the validation and test splits of the data
    language_input = []
    visual_input = []
    targets = []

    for img in batch:
        vggs = torch.load("/home/u924823/ha_bbox_vggs/"+img+".pt").to(device)
        for obj in dict_words_boxes[img]:
            language_input.append(get_word_ix(word_to_ix, dict_words_boxes[img][obj]["word"]))

            bbox_indices = []
            n = 0

            for obj_id in dict_words_boxes[img]:
                bbox_indices.append(ha_vggs_indices[img][obj_id][0])
                if obj_id == obj:
                    targets.append(n)
                n += 1
            visual_input.append(vggs[bbox_indices,:])

    lang_batch = torch.tensor(language_input, dtype=torch.long, device=device)
    vis_batch = torch.stack(visual_input)
    targets = torch.tensor(targets, dtype=torch.long, device=device)
    return lang_batch, vis_batch, targets


def load_img(dict_words_boxes, ha_vggs_indices, img):
    vggs = torch.load("/home/u924823/ha_bbox_vggs/"+img+".pt").to(device) # Edit path
    # dict met obj ids als keys en een dictionary met words : '', bboxes :
    n = 0
    bbox_indices = []
    words = []
    for obj in dict_words_boxes[img]: # For every object in this image
        words.append(get_word_ix(word_to_ix, dict_words_boxes[img][obj]["word"]))
        bbox_indices.append(ha_vggs_indices[img][obj][0])
    visual_input = vggs[bbox_indices,:]
    language_input = torch.tensor(words, device=device)
    return language_input, visual_input


def random_look_at_img(dict_words_boxes, ha_vggs_indices, img):
    language_input, scene = load_img(dict_words_boxes, ha_vggs_indices, img)
    # repeat scene n_objects times as input to listener
    visual_input = scene.expand(scene.size()[0], scene.size()[0], scene.size()[1])
    #targets = torch.eye(visual_input.size()[0], dtype=torch.long, device=device)
    targets = torch.tensor([i for i in range(len(language_input))], dtype=torch.long, device=device)

    i = np.random.randint(len(targets))
    return language_input[i], scene, targets[i]

def load_select_obj(dict_words_boxes, ha_vggs_indices, img, setting):
    if setting == "random":
        return random_look_at_img(dict_words_boxes, ha_vggs_indices, img)


def calc_accuracy(guesses, targets, average=True):
    """
    in: log probabilities for C classes (i.e. candidate nrs), target 'class'
    indices (from 0 up-to-and-icluding C-1) (object position in your case)
    """
    score = 0
    guess = torch.argmax(guesses.data, 1)

    for i in range(targets.data.size()[0]):
        if guess.data[i] == targets.data[i]:
            score += 1
    
    if average:
        return score/targets.data.size()[0], targets.data.size()[0]
    else:
        return score, targets.data.size()[0]
  
    
def targets_1_1(language_input, sibling_inputs):
    """
    input:targets and word guesses from sibling as target to child
    output: returns a torch tensor like targets but every each value replaced 
            word guesses from sibling
    """
    targets_child = []
    for i in range(0,len(language_input)):
        if i % 2 == 0:
            targets_child.append(language_input[i])
        elif i % 2 == 1:
            targets_child.append(sibling_inputs[i])
    t = np.array(targets_child,dtype='int')
    return torch.from_numpy(t)


def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes) 
    return y[labels] 

    
################## TRAINING FUNCTION ##########################################
        

def train():
    listener_child.train()
    speaker_child.train()
#    listener_sibling.train()
    speaker_sibling.train()
    
    start_time = time.time()
    n_batches = len(batches)
    
    # for sibling metrics
#    li_train_sibling_loss = np.empty(n_batches)
#    li_train_sibling_accuracy = np.empty(n_batches)
    sp_train_sibling_loss = np.empty(n_batches)
    sp_train_sibling_accuracy = np.empty(n_batches)
    
    # for child metrics
    li_train_child_loss = np.empty(n_batches)
    li_train_child_accuracy = np.empty(n_batches)
    sp_train_child_loss = np.empty(n_batches)
    sp_train_child_accuracy = np.empty(n_batches)
    
    batch_size = np.empty(n_batches)

    batch = 0
    
    # batches shuffled during training
    while batch < n_batches:
        language_batch = [] # All word indices in the batch?
        visual_batch = [] # All vgg vectors in the batch?
        target_batch = [] # All target word indices in the batch?

        for img in batches[batch]:
            language_input, visual_input, target = load_select_obj(dict_words_boxes, ha_vggs_indices, img,
                setting)
            language_batch.append(language_input)
            visual_batch.append(visual_input)
            target_batch.append(target)
        language_input = torch.stack(language_batch)
        visual_input = torch.stack(visual_batch)
        targets = torch.stack(target_batch)

        speaker_child_optimizer.zero_grad()
        listener_child_optimizer.zero_grad()

################ sibling training #############################################

        # obj_gueses of sibling
#        obj_guess_values_sibling = torch.nn.functional.one_hot(language_input)
        

        obj_guess_values_sibling = one_hot_embedding(targets, visual_input.shape[1]).to(device)
#        obj_guesses_sibling = listener_sibling(language_input, visual_input)

        # Saves the batch length for weighted mean accuracy:
        batch_size[batch] = len(batches[batch])

#        listener_sibling_loss = criterion(obj_guesses_sibling, targets)
#        listener_sibling_loss.backward() # backward pass
#        listener_sibling_optimizer.step() # adapting the weights

        # Loss/accuracy times batch size for weighted average over epoch:
#        li_train_sibling_loss[batch] = listener_sibling_loss.item() * batch_size[batch]
#        li_train_sibling_accuracy[batch],_ = calc_accuracy(obj_guesses_sibling, targets, average=False)
#
#        obj_guess_values_sibling = obj_guesses_sibling.detach()

        word_guesses_sibling = speaker_sibling(visual_input, obj_guess_values_sibling)

        speaker_sibling_loss = criterion(word_guesses_sibling, language_input)
        speaker_sibling_loss.backward()
        speaker_sibling_optimizer.step()
        
        sibling_language_output = torch.argmax(word_guesses_sibling ,dim =1)
        
        # Loss/accuracy times batch size for weighted average over epoch:
        sp_train_sibling_loss[batch] = speaker_sibling_loss.item() * batch_size[batch]
        sp_train_sibling_accuracy[batch],_ = calc_accuracy(word_guesses_sibling, language_input, average=False)
        
        language_input_child = targets_1_1(language_input, sibling_language_output).to(device)
    
        
################ child training ###############################################
                
        # obj_guesses of child        
        obj_guesses_child = listener_child(language_input_child, visual_input)

        # Saves the batch length for weighted mean accuracy:
        batch_size[batch] = len(batches[batch])
        
        listener_child_loss = criterion(obj_guesses_child, targets)
        listener_child_loss.backward() # backward pass
        listener_child_optimizer.step() # adapting the weights
        
        # Loss/accuracy times batch size for weighted average over epoch:
        li_train_child_loss[batch] = listener_child_loss.item() * batch_size[batch]
        li_train_child_accuracy[batch],_ = calc_accuracy(obj_guesses_child, targets, average=False)
        
        obj_guess_values_child = obj_guesses_child.detach()
        
        word_guesses_child = speaker_child(visual_input, obj_guess_values_child)
        
        speaker_child_loss = criterion(word_guesses_child, language_input)
        speaker_child_loss.backward()
        speaker_child_optimizer.step()
                
        # Loss/accuracy times batch size for weighted average over epoch:
        sp_train_child_loss[batch] = speaker_child_loss.item() * batch_size[batch]
        sp_train_child_accuracy[batch],_ = calc_accuracy(word_guesses_child, language_input, average=False)
        
        
        batch += 1
        if batch % printerval == 0:
            print('| epoch {:2d} | batch {:3d}/{:3d} | t {:6.2f} | s.S.L {:6.4f} | s.S.A {:5.4f} | l.C.L {:6.4f} | l.C.A {:5.4f} | s.C.L {:6.4f} | s.C.A {:5.4f} |'.format(
                epoch, batch, n_batches, (time.time() - start_time),
#                np.sum(li_train_sibling_loss[batch-printerval:batch])/np.sum(batch_size[batch-printerval:batch]),
#                np.sum(li_train_sibling_accuracy[batch-printerval:batch])/np.sum(batch_size[batch-printerval:batch]),
                np.sum(sp_train_sibling_loss[batch-printerval:batch])/np.sum(batch_size[batch-printerval:batch]),
                np.sum(sp_train_sibling_accuracy[batch-printerval:batch])/np.sum(batch_size[batch-printerval:batch]),
                
                np.sum(li_train_child_loss[batch-printerval:batch])/np.sum(batch_size[batch-printerval:batch]),
                np.sum(li_train_child_accuracy[batch-printerval:batch])/np.sum(batch_size[batch-printerval:batch]),
                np.sum(sp_train_child_loss[batch-printerval:batch])/np.sum(batch_size[batch-printerval:batch]),
                np.sum(sp_train_child_accuracy[batch-printerval:batch])/np.sum(batch_size[batch-printerval:batch])))
    
#    avg_li_train_sibling_loss = np.sum(li_train_sibling_loss) / np.sum(batch_size)
#    avg_li_train_sibling_acc = np.sum(li_train_sibling_accuracy) / np.sum(batch_size)
    avg_sp_train_sibling_loss = np.sum(sp_train_sibling_loss) / np.sum(batch_size)
    avg_sp_train_sibling_acc = np.sum(sp_train_sibling_accuracy) / np.sum(batch_size)
    
    
    avg_li_train_child_loss = np.sum(li_train_child_loss) / np.sum(batch_size)
    avg_li_train_child_acc = np.sum(li_train_child_accuracy) / np.sum(batch_size)
    avg_sp_train_child_loss = np.sum(sp_train_child_loss) / np.sum(batch_size)
    avg_sp_train_child_acc = np.sum(sp_train_child_accuracy) / np.sum(batch_size)

    print('-' * 89)
    print("overall performance on training set:")
    print('| L.C.loss {:8.4f} | L.C.acc. {:8.4f} |'.format(
#        avg_li_train_sibling_loss,
#        avg_li_train_sibling_acc,
        avg_li_train_child_loss,
        avg_li_train_child_acc))
    print('| S.S.loss {:8.4f} | S.S.acc. {:8.4f} | S.C.loss {:8.4f} | S.C.acc. {:8.4f} |'.format(
        avg_sp_train_sibling_loss,
        avg_sp_train_sibling_acc,
        avg_sp_train_child_loss,
        avg_sp_train_child_acc))
    print('-' * 89)
    return avg_li_train_child_loss, avg_li_train_child_acc, avg_sp_train_sibling_loss, avg_sp_train_sibling_acc, avg_sp_train_child_loss, avg_sp_train_child_acc



################ Interpret command line arguments #############################
print("\nSys.argv:",sys.argv)
batchsize = int(sys.argv[1])
lr = float(sys.argv[2])
setting = "random"


################## LOADING DATA ###############################################

# Object vgg indices (object information)
with open("/home/u924823/data/ha_vgg_indices.json", "rb") as input_file:
    ha_vggs_indices = json.load(input_file)
    
# Regular data (dictionary with all images and their object ids, corresponding words)
with open("/home/u924823/data/dict_words_boxes.json", "rb") as input_file:
    dict_words_boxes = json.load(input_file)

# Train split, image ids
with open("/home/u924823/data/train_data.txt", "rb") as fp:
    train_data = pickle.load(fp)

# Validation split, image ids
with open("/home/u924823/data/validation_data.txt", "rb") as fp:
    validation_data = pickle.load(fp)

# Test split, image ids
with open("/home/u924823/data/test_data.txt", "rb") as fp:
    test_data = pickle.load(fp)


################# PREPROCESSING THE LOADED DATA ###############################  
    
    

# Makes a vocabulary of the entire set of objects:
vocab, freq = make_vocab(dict_words_boxes)

# Gives an index number to every word in the vocabulary:
word_to_ix = make_ix_table(vocab)

# Returns a dictionary with the number of objects per image:
no_objs = no_of_objs(dict_words_boxes, train_data)

# Returns a list of batch-size batches:
# A batch contains images with the same no. of objs
batches = dict_to_batches(no_objs, batchsize)

no_objs_val = no_of_objs(dict_words_boxes, validation_data)
val_batchlist = dict_to_batches(no_objs_val, batchsize)
#test set:
no_objs_test = no_of_objs(dict_words_boxes, test_data)
test_batchlist = dict_to_batches(no_objs_test, batchsize)

ntokens = len(word_to_ix.keys())
print("ntokens:",ntokens)



################### SPECIFY  CHILD MODEL #############################################


# these are the sizes Anna Rohrbach uses. she uses a batch size of 40.
n_objects = 100
object_size = 4096 # Length vgg vector?
att_hidden_size = 256 # Number of hidden nodes
wordemb_size = 256 # Length word embedding

print("hidden layer size:", att_hidden_size)

epochs =  50 # start with 1 while testing

device = torch.device('cuda') # Device = GPU

# Makes the listener part of the child:
listener_child = child_agent.Listener(n_objects, object_size, ntokens, wordemb_size,
        att_hidden_size).to(device)

# Makes the speaker part of the child:
speaker_child = child_agent.Speaker(object_size, ntokens, att_hidden_size).to(device)

# Loss function: binary cross entropy
criterion = nn.CrossEntropyLoss(size_average = True)



############ LOAD SIBLING ######################################

# Loads the listener part of the sibling:
#listener_sibling = child_agent.Listener(n_objects, object_size, ntokens, wordemb_size,
#        att_hidden_size).to(device)
#listener_sibling_optimizer = optim.Adam(listener_sibling.parameters(), lr=lr)

#checkpoint = torch.load('/home/u924823/dyadic_scenario/pretrained_sibling_models/{}_{}_{}_checkpoint_sibling_listener.pth'.format(str(batchsize),lr,str(epochs)))
#listener_sibling.load_state_dict(checkpoint['model_state_dict'])
#listener_sibling_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#epoch = checkpoint['epoch']
#listener_sibling_loss = checkpoint['loss']

# Loads the speaker part of the sibling:

speaker_sibling = child_agent.Speaker(object_size, ntokens, att_hidden_size).to(device)
speaker_sibling_optimizer = optim.Adam(speaker_sibling.parameters(), lr=lr)

checkpoint = torch.load('/home/u924823/dyadic_scenario/pretrained_sibling_models/{}_{}_{}_checkpoint_sibling_speaker.pth'.format(str(batchsize),lr,str(epochs)))
speaker_sibling.load_state_dict(checkpoint['model_state_dict'])
speaker_sibling_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])





###############################################################################
# TRAIN LOOP
###############################################################################
# Print after this many batches:
printerval = 100

print("parameters of listener child:")
for param in listener_child.parameters():
    print(type(param.data), param.size())
listener_child_optimizer = optim.Adam(listener_child.parameters(), lr=lr)

print("parameters of speaker child:")
for param in speaker_child.parameters():
    print(type(param.data), param.size())
speaker_child_optimizer = optim.Adam(speaker_child.parameters(), lr=lr)


# Creating numpy arrays to store loss and accuracy
# for train, validation, and test splits -- SIBLING
#listener_train_sibling_loss = np.empty(epochs)
#listener_train_sibling_acc = np.empty(epochs)
speaker_train_sibling_loss = np.empty(epochs)
speaker_train_sibling_acc = np.empty(epochs)
#listener_val_sibling_loss = np.empty(epochs)
#listener_val_sibling_acc = np.empty(epochs)
speaker_val_sibling_loss = np.empty(epochs)
speaker_val_sibling_acc = np.empty(epochs)
#listener_test_sibling_loss = np.empty(epochs)
#listener_test_sibling_acc = np.empty(epochs)
speaker_test_sibling_loss = np.empty(epochs)
speaker_test_sibling_acc = np.empty(epochs)



# Creating numpy arrays to store loss and accuracy
# for train, validation, and test splits -- CHILD
listener_train_child_loss = np.empty(epochs)
listener_train_child_acc = np.empty(epochs)
speaker_train_child_loss = np.empty(epochs)
speaker_train_child_acc = np.empty(epochs)
listener_val_child_loss = np.empty(epochs)
listener_val_child_acc = np.empty(epochs)
speaker_val_child_loss = np.empty(epochs)
speaker_val_child_acc = np.empty(epochs)
listener_test_child_loss = np.empty(epochs)
listener_test_child_acc = np.empty(epochs)
speaker_test_child_loss = np.empty(epochs)
speaker_test_child_acc = np.empty(epochs)

# At any point you can hit Ctrl + C to break out of training early.

            
           

try:
    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()
        
        li_train_child_loss, li_train_child_acc, sp_train_sibling_loss, sp_train_sibling_acc, sp_train_child_loss, sp_train_child_acc = train()
        
#        li_train_loss, li_train_acc, sp_train_loss, sp_train_acc = train()
        
#        listener_train_sibling_loss[epoch-1], listener_train_sibling_acc[epoch-1] = li_train_sibling_loss, li_train_sibling_acc #sibling
        speaker_train_sibling_loss[epoch-1], speaker_train_sibling_acc[epoch-1] = sp_train_sibling_loss, sp_train_sibling_acc #sibling
        listener_train_child_loss[epoch-1], listener_train_child_acc[epoch-1] = li_train_child_loss, li_train_child_acc  #child
        speaker_train_child_loss[epoch-1], speaker_train_child_acc[epoch-1] = sp_train_child_loss, sp_train_child_acc  #child
        
        
        li_val_child_loss, li_val_child_acc, sp_val_sibling_loss, sp_val_sibling_acc, sp_val_child_loss, sp_val_child_acc = evaluate(epoch)
        
#        listener_val_sibling_loss[epoch-1], listener_val_sibling_acc[epoch-1] = li_val_sibling_loss, li_val_sibling_acc
        speaker_val_sibling_loss[epoch-1], speaker_val_sibling_acc[epoch-1] = sp_val_sibling_loss, sp_val_sibling_acc
        listener_val_child_loss[epoch-1], listener_val_child_acc[epoch-1] = li_val_child_loss, li_val_child_acc
        speaker_val_child_loss[epoch-1], speaker_val_child_acc[epoch-1] = sp_val_child_loss, sp_val_child_acc
        
#        li_test_loss, li_test_acc, sp_test_loss, sp_test_acc 
        li_test_child_loss, li_test_child_acc, sp_test_sibling_loss, sp_test_sibling_acc, sp_test_child_loss, sp_test_child_acc= evaluate(epoch,'test')
        
#        listener_test_sibling_loss[epoch-1], listener_test_sibling_acc[epoch-1] = li_test_sibling_loss, li_test_sibling_acc
        speaker_test_sibling_loss[epoch-1], speaker_test_sibling_acc[epoch-1] = sp_test_sibling_loss, sp_test_sibling_acc
        listener_test_child_loss[epoch-1], listener_test_child_acc[epoch-1] = li_test_child_loss, li_test_child_acc
        speaker_test_child_loss[epoch-1], speaker_test_child_acc[epoch-1] = sp_test_child_loss, sp_test_child_acc
        
# To enable to hit Ctrl + C and break out of training:
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Saving the loss and accuracy numpy arrays - SIBLING:
#np.save('loss_acc/li_train_sibling_loss_{}_{}'.format(str(batchsize),
#    str(lr)), listener_train_sibling_loss)
#np.save('loss_acc/li_train_sibling_acc_{}_{}'.format(str(batchsize),
#    str(lr)), listener_train_sibling_acc)
np.save('loss_acc/sp_train_sibling_loss_{}_{}'.format(str(batchsize),
    str(lr)), speaker_train_sibling_loss)
np.save('loss_acc/sp_train_sibling_acc_{}_{}'.format(str(batchsize),
    str(lr)), speaker_train_sibling_acc)

#np.save('loss_acc/li_val_sibling_loss_{}_{}'.format(str(batchsize),
#    str(lr)), listener_val_sibling_loss)
#np.save('loss_acc/li_val_sibling_acc_{}_{}'.format(str(batchsize),
#    str(lr)), listener_val_sibling_acc)
np.save('loss_acc/sp_val_sibling_loss_{}_{}'.format(str(batchsize),
    str(lr)), speaker_val_sibling_loss)
np.save('loss_acc/sp_val_sibling_acc_{}_{}'.format(str(batchsize),
    str(lr)), speaker_val_sibling_acc)

#np.save('loss_acc/li_test_sibling_loss_{}_{}'.format(str(batchsize),
#    str(lr)), listener_test_sibling_loss)
#np.save('loss_acc/li_test_sibling_acc_{}_{}'.format(str(batchsize),
#    str(lr)), listener_test_sibling_acc)
np.save('loss_acc/sp_test_sibling_loss_{}_{}'.format(str(batchsize),
    str(lr)), speaker_test_sibling_loss)
np.save('loss_acc/sp_test_sibling_acc_{}_{}'.format(str(batchsize),
    str(lr)), speaker_test_sibling_acc)

# Saving the loss and accuracy numpy arrays - CHILD:
np.save('loss_acc/li_train_child_loss_{}_{}'.format(str(batchsize),
    str(lr)), listener_train_child_loss)
np.save('loss_acc/li_train_child_acc_{}_{}'.format(str(batchsize),
    str(lr)), listener_train_child_acc)
np.save('loss_acc/sp_train_child_loss_{}_{}'.format(str(batchsize),
    str(lr)), speaker_train_child_loss)
np.save('loss_acc/sp_train_child_acc_{}_{}'.format(str(batchsize),
    str(lr)), speaker_train_child_acc)

np.save('loss_acc/li_val_child_loss_{}_{}'.format(str(batchsize),
    str(lr)), listener_val_child_loss)
np.save('loss_acc/li_val_child_acc_{}_{}'.format(str(batchsize),
    str(lr)), listener_val_child_acc)
np.save('loss_acc/sp_val_child_loss_{}_{}'.format(str(batchsize),
    str(lr)), speaker_val_child_loss)
np.save('loss_acc/sp_val_child_acc_{}_{}'.format(str(batchsize),
    str(lr)), speaker_val_child_acc)

np.save('loss_acc/li_test_child_loss_{}_{}'.format(str(batchsize),
    str(lr)), listener_test_child_loss)
np.save('loss_acc/li_test_child_acc_{}_{}'.format(str(batchsize),
    str(lr)), listener_test_child_acc)
np.save('loss_acc/sp_test_child_loss_{}_{}'.format(str(batchsize),
    str(lr)), speaker_test_child_loss)
np.save('loss_acc/sp_test_child_acc_{}_{}'.format(str(batchsize),
    str(lr)), speaker_test_child_acc)








