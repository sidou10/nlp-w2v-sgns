import numpy as np
from math import exp, log
from tqdm import tqdm_notebook
from scipy.special import expit
import random


def log_sigmoid(x):
    return log(expit(x))

def update_voc(voc_dict, word, index):
    # (word: index)
    if word not in voc_dict:
        voc_dict[word] = index
        index += 1
    return voc_dict, index


def getPairsAndVocs(sentences, winSize):
    wc_pairs = []
    word_voc = {}
    word_index = 0
    context_voc = {}
    context_index = 0
    
    for sent in sentences:
        for i in range(len(sent)):
            word = sent[i]
            # Update word_voc dict
            word_voc, word_index = update_voc(word_voc, word, word_index)
            for j in range(max(0,i-winSize), min(i+winSize+1, len(sent))):
                if i != j:
                    context = sent[j]
                    context_voc, context_index = update_voc(context_voc, context, context_index)
                    wc_pairs.append((word_voc[word], context_voc[context]))
    
    random.shuffle(wc_pairs)    
    return wc_pairs, word_voc, context_voc

def getNegPairs(wc_pairs, word_voc, context_voc, negativeRate):
    word_voc_list = list(word_voc.keys())
    context_voc_list = list(context_voc.keys())
    
    nb_words = len(word_voc_list)
    nb_contexts = len(context_voc_list)
    
    nb_pairs = len(wc_pairs)
    neg_wc_pairs = []

    c = 0
    while c < negativeRate*nb_pairs:
        ind1 = np.random.randint(0,nb_words)
        ind2 = np.random.randint(0,nb_contexts)
        neg_wc_pairs.append((ind1, ind2))
        c += 1
    
    random.shuffle(neg_wc_pairs)
    return neg_wc_pairs
    
def costFunction(theta, nEmbed, wc_pairs, neg_wc_pairs, nb_words, nb_contexts):
    W = theta[:nEmbed*nb_words].reshape(nb_words, nEmbed)
    C = theta[nEmbed*nb_words:].reshape(nb_contexts, nEmbed)
    
    wc_cost = sum([log_sigmoid(W[wc_pair[0],:].dot(C[wc_pair[1],:])) for wc_pair in wc_pairs])
    neg_wc_cost = sum([log_sigmoid(-W[neg_wc_pair[0],:].dot(C[neg_wc_pair[1],:])) for neg_wc_pair in neg_wc_pairs])
        
    return -wc_cost-neg_wc_cost

def gradCost(theta, nEmbed, wc_pairs, neg_wc_pairs, nb_words, nb_contexts):
    grad = np.zeros(theta.shape[0])

    W = theta[:nEmbed*nb_words].reshape(nb_words, nEmbed)
    C = theta[nEmbed*nb_words:].reshape(nb_contexts, nEmbed)
    
    for wc_pair in wc_pairs:
        word_idx = wc_pair[0]
        context_idx = wc_pair[1]

        word = W[word_idx,:]
        context = C[context_idx,:]

        exp_mwdotc = exp(-word.dot(context))
        
        # Update the derivative for word and context
        df_dw = context*exp_mwdotc/(1+exp_mwdotc)
        grad[word_idx*nEmbed:(word_idx+1)*nEmbed] += df_dw

        df_dc = word*exp_mwdotc/(1+exp_mwdotc)
        grad[context_idx*nEmbed:(context_idx+1)*nEmbed] += df_dc

    for neg_wc_pair in neg_wc_pairs:
        word_idx = neg_wc_pair[0]
        context_idx = neg_wc_pair[1]

        word = W[word_idx,:]
        context = C[context_idx,:]

        exp_wdotc = exp(word.dot(context))
        
        # Update the derivative for word and context
        df_dw = -context*exp_wdotc/(1+exp_wdotc)
        grad[word_idx*nEmbed:(word_idx+1)*nEmbed] += df_dw
        
        df_dc = -word*exp_wdotc/(1+exp_wdotc)
        grad[context_idx*nEmbed:(context_idx+1)*nEmbed] += df_dc
    
    return grad