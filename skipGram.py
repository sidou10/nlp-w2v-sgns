from __future__ import division
import argparse
import pandas as pd

# useful stuff
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import normalize


__authors__ = ['author1','author2','author3']
__emails__  = ['fatherchristmoas@northpole.dk','toothfairy@blackforest.no','easterbunny@greenfield.de']

def text2sentences(path):
    # feel free to make a better tokenization/pre-processing
    sentences = []
    with open(path) as f:
        for l in f:
            sentences.append( l.lower().split() )
    return sentences

def loadPairs(path):
    data = pd.read_csv(path,delimiter='\t')
    pairs = zip(data['word1'],data['word2'],data['similarity'])
    return pairs

def update_voc(voc_dict, word, nEmbed):
    # (word: [count, embedding])
    if word in voc_dict:
        voc_dict[word][0] += 1
    else:
        voc_dict[word] = [1, np.ones(nEmbed)*np.random.random()]
    return voc_dict

def getPairsAndVocs(sentences, winSize, nEmbed):
    wc_pairs = []
    word_voc = {}
    context_voc = {}
    for sent in sentences:
        for i in range(len(sent)):
            word = sent[i]
            # Update word_voc dict
            word_voc = update_voc(word_voc, word, nEmbed)
            for j in range(max(0,i-winSize), min(i+winSize+1, len(sent))):
                if i != j:
                    context = sent[j]
                    context_voc = update_voc(context_voc, context, nEmbed)
                    wc_pairs.append((word, context))
    return wc_pairs, word_voc, context_voc

def getNegPairs(wc_pairs, word_voc):
    word_voc_list = list(word_voc.keys())
    nb_words = len(word_voc_list)
    nb_pairs = len(wc_pairs)
    for _ in range(nb_pairs):
        ind1 = np.random.randint(0,nb_words)
        ind2 = np.random.randint(0,nb_words)
        neg_wc_pairs.append((word_voc_list[ind1], word_voc_list[ind2]))
    return neg_wc_pairs

class mSkipGram:
    def __init__(self,sentences, nEmbed=100, negativeRate=5, winSize=5, minCount=5):
        self.wc_pairs, self.word_voc, self.context_voc = getPairsAndVocs(sentences, winSize, nEmbed)
        self.neg_wc_pairs = getNegPairs(self.wc_pairs, self.word_voc)

    def train(self, stepsize, epochs):
        raise NotImplementedError('implement it!')

    def save(self, path):
        raise NotImplementedError('implement it!')

    def similarity(self,word1,word2):
        """
            computes similiarity between the two words. unknown words are mapped to one common vector
        :param word1:
        :param word2:
        :return: a float \in [0,1] indicating the similarity (the higher the more similar)
        """
        raise NotImplementedError('implement it!')

    @staticmethod
    def load(path):
        raise NotImplementedError('implement it!')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')

    opts = parser.parse_args()

    if not opts.test:
        sentences = text2sentences(opts.text)
        sg = mySkipGram.gSkipGram(sentences)
        sg.train()
        sg.save(opts.model)

    else:
        pairs = loadPairs(opts.text)

        sg = mSkipGram.load(opts.model)
        for a,b,_ in pairs:
            print sg.similarity(a,b)

