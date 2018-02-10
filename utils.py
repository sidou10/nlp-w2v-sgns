import numpy as np
from math import exp, log
from tqdm import tqdm_notebook
from scipy.special import expit
from nltk.probability import FreqDist
import random


def log_sigmoid(x):
    try:
        return log(expit(x))
    except ValueError:
        print("x:{}".format(x))

def update_voc(voc_dict, word, index):
    # (word: index, count)
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
    
    np.random.shuffle(wc_pairs)
    return wc_pairs, word_voc, context_voc

def getNegPairs(wc_pairs, word_voc, context_voc, negativeRate):
    word_voc_list = list(word_voc.keys())
    context_voc_list = list(context_voc.keys())
    
    nb_words = len(word_voc_list)
    nb_contexts = len(context_voc_list)
    
    nb_pairs = len(wc_pairs)
    neg_wc_pairs = []

    for wc_pair in wc_pairs:
        word = wc_pair[0]
        for _ in range(negativeRate):
            context_idx = np.random.randint(nb_pairs)
            context = wc_pairs[context_idx][1]
            neg_wc_pairs.append((word,context))

    np.random.shuffle(neg_wc_pairs)
    return neg_wc_pairs
    
def costFunction(theta, nEmbed, wc_pairs, neg_wc_pairs, nb_words, nb_contexts):
    W = theta[:nEmbed*nb_words].reshape(nb_words, nEmbed)
    C = theta[nEmbed*nb_words:].reshape(nb_contexts, nEmbed)
    
    wc_cost = sum([log_sigmoid(W[wc_pair[0]].dot(C[wc_pair[1]])) for wc_pair in wc_pairs])
    neg_wc_cost = sum([log_sigmoid(-W[neg_wc_pair[0]].dot(C[neg_wc_pair[1]])) for neg_wc_pair in neg_wc_pairs])
        
    return -wc_cost-neg_wc_cost
    #return -wc_cost

def gradCost(theta, nEmbed, wc_pairs, neg_wc_pairs, nb_words, nb_contexts):
    grad = np.zeros(theta.shape[0])

    W = theta[:nEmbed*nb_words].reshape(nb_words, nEmbed)
    C = theta[nEmbed*nb_words:].reshape(nb_contexts, nEmbed)
    
    for wc_pair in wc_pairs:
        word_idx = wc_pair[0]
        context_idx = wc_pair[1]

        word = W[word_idx]
        context = C[context_idx]

        exp_mwdotc = exp(-word.dot(context))
        
        # Update the derivative for word and context
        df_dw = context*exp_mwdotc/(1+exp_mwdotc)
        grad[word_idx*nEmbed:(word_idx+1)*nEmbed] += df_dw

        df_dc = word*exp_mwdotc/(1+exp_mwdotc)
        grad[(nb_words+context_idx)*nEmbed:(nb_words+context_idx+1)*nEmbed] += df_dc

    for neg_wc_pair in neg_wc_pairs:
        word_idx = neg_wc_pair[0]
        context_idx = neg_wc_pair[1]

        word = W[word_idx]
        context = C[context_idx]

        exp_wdotc = exp(word.dot(context))
        
        # Update the derivative for word and context
        df_dw = -context*exp_wdotc/(1+exp_wdotc)
        grad[word_idx*nEmbed:(word_idx+1)*nEmbed] += df_dw
        
        df_dc = -word*exp_wdotc/(1+exp_wdotc)
        grad[(nb_words+context_idx)*nEmbed:(nb_words+context_idx+1)*nEmbed] += df_dc
    
    return grad

def remove_rare_words(sentences, minCount):
    fd = FreqDist([word for sent in sentences for word in sent])

    sentences_w_min_count = []
    for sent in sentences:
        sent_w_min_count = []
        for word in sent:
            if fd[word] > minCount:
                sent_w_min_count.append(word)
        sentences_w_min_count.append(sent_w_min_count)

    return sentences_w_min_count

