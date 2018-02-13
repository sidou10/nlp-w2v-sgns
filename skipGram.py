from __future__ import division
import argparse
import pandas as pd

# useful stuff
import re
import time
from math import exp, log
import pickle
import string
import numpy as np


__authors__ = ['Amine Sekkat','Lina El Azhari','Nabil Toumi', 'Saad Ben Cherif']
__emails__  = ['mohamed-amine.sekkat@student.ecp.fr','lina.el-azhari@student.ecp.fr','nabil.toumi@student.ecp.fr', 'saad.ben-cherif-ouedrhiri@student.ecp.fr']


## FUNCTIONS RAW TEXT -> SENTENCES 

'''
Initial text2sentences function. Instead of considering one line as a sentence,
we first concatenate all the raw document and split it by ".?!".
We then remove the punctuation and tokenize each sentence.
'''

def concat_text(path):
    '''
    Concatenates all the text in a file located at file path
    '''
    all_text_list = []
    with open(path) as f:
        for l in f:
            l = l[:-1] #We remove the line jumping
            all_text_list.append(l)
    concat_text = ''.join(all_text_list)
    return concat_text

def text2sentences(path):
    '''
    Converts a raw text from path to tokenized sentences 
    '''
    concatenated_text = concat_text(path)
    # Split at each .?!
    sentences = re.split(r'[.?!]', concatenated_text)
    # Remove punctuation
    reg = re.compile('[%s]' % re.escape(string.punctuation))
    # Tokenize
    sentences_token = []
    for sent in sentences:
        sent_wo_punct = reg.sub('', sent)
        sentences_token.append( sent_wo_punct.lower().split() )

    return sentences_token

def loadPairs(path):
    data = pd.read_csv(path,delimiter='\t')
    pairs = zip(data['word1'],data['word2'],data['similarity'])
    return pairs

# UTILS
def log_sigmoid(x):
    return -log(1+exp(-x))

def freq_dist(words):
    '''
    Returns a dictionnary with the count of each word in the words list
    '''
    fd = {}
    for word in words:
        if word in fd:
            fd[word] += 1
        else:
            fd[word] = 1
    return fd

class DivergentGradientError(Exception):
    '''
    Raised when the values of the gradient explode
    '''
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class mySkipGram:
    def __init__(self, sentences, nEmbed=100, negativeRate=5, winSize=3, minCount=5):
        # Store embedding dimension and negative rate for ulterior usage
        print("Parameters: nEmbed={}, negativeRate={}, winSize={}, minCount={}".format(nEmbed, negativeRate, winSize, minCount))
        self.nEmbed = nEmbed
        self.negativeRate = negativeRate

        # PREPROCESS SENTENCES
        # Remove rare words (i.e. words that appear less than minCount)
        sentences_w_min_count = self.remove_rare_words(sentences, minCount)
        # Subsample (check subsample function)
        #sentences_sbsmpld = subsample(sentences_w_min_count, 10)

        # Through a single pass in the sentences list, we create
        # - a list of (word, context) pairs
        # - a dictionnary keeping track of each word and an associated index
        # - a dictionnary keeping track of each context and an associated index
        print("Generating training set and vocabularies...", end=" ")
        self.wc_pairs, self.word_voc, self.context_voc = self.get_pairs_and_vocs(sentences_w_min_count, winSize)
        print("ok!\nGenerating negative samples...", end=" ")
        # Generate negative samples
        self.neg_wc_pairs = self.get_neg_pairs(self.wc_pairs, self.word_voc, self.context_voc, negativeRate)
        print("ok!")


    def train(self, stepsize=0.01, epochs=5, batchsize=512):
        '''
        :param stepsize, step size is SGD:
        :param epochs, nb of times all training set is viewed:
        :param batchsize, nb of positive samples considered at each iteration of SGD:
        Creates a W matrix containing the embeddings of all words by 
        '''

        # Get nb of words, nb of contexts, nb of pairs
        nb_words = len(list(self.word_voc.keys()))
        nb_contexts = len(list(self.context_voc.keys()))
        nb_pairs = len(self.wc_pairs)

        # Get nb of parameters
        nb_param = self.nEmbed*(nb_words+nb_contexts)
        
        # Initialize vector of parameters to small values
        theta = np.random.random(nb_param)*1e-5

        # This function computes the gradient at a point theta for 
        # some positive and negative samples
        grad = lambda theta, wc_pairs, neg_wc_pairs: self.grad_cost_function(theta, self.nEmbed, wc_pairs, neg_wc_pairs, nb_words, nb_contexts)

        ### STOCHASTIC GRADIENT DESCENT ###
        print("TRAINING: #epochs: {}, step size: {}, batch size: {}".format(epochs, stepsize, batchsize))
        for epoch in range(epochs):
            print("Epoch {}/{}".format(epoch+1, epochs))
            t1 = time.time()
            for batch in range(nb_pairs//batchsize):
                batch_begin = batch*batchsize
                batch_end = (batch+1)*batchsize
                if batch_end > nb_pairs:
                    batch_end = nb_pairs
                batch_pos = self.wc_pairs[batch_begin:batch_end]
                batch_neg = self.neg_wc_pairs[self.negativeRate*batch_begin:self.negativeRate*batch_end]
                gradn = grad(theta, batch_pos, batch_neg)
                theta = theta - stepsize*gradn
            t2 = time.time()
            print("({:.2f}s)".format(t2-t1))
        ### STOCHASTIC GRADIENT DESCENT ###
        
        self.theta = theta
        self.W = theta[:self.nEmbed*nb_words].reshape(nb_words, self.nEmbed)

    def save(self,path):
        '''
        Save W (matrix of embeddings) and word_voc (dictionnary of (word: index)) in path
        '''
        with open(path, 'wb') as f:
            pickle.dump([self.W, self.word_voc], f)
        
    @staticmethod
    def similarity(word1,word2,W,word_voc):
        """
        computes similiarity between the two words. unknown words are mapped to one common vector
        :param word1:
        :param word2:
        :param W, array of size (nEmbed,nb_words) embedding of word i available at W[i]:
        :param word_voc, dict of word: index
        :return: a float \in [0,1] indicating the similarity (the higher the more similar)
        """
        nEmbed = W.shape[1]
        default_embd = np.ones(nEmbed)*0.01

        if word1 in word_voc:
            idx_word1 = word_voc[word1]
            embd_word1 = W[idx_word1]
        else:
            print("Unknown word '{}' mapped to default embedding (0.1)".format(word1))
            embd_word1 = default_embd

        if word2 in word_voc:
            idx_word2 = word_voc[word2]
            embd_word2 = W[idx_word2]
        else:
            print("Unknown word '{}' mapped to default embedding (0.1)".format(word2))
            embd_word2 = default_embd

        cosine = embd_word1.dot(embd_word2)/(np.linalg.norm(embd_word1)*np.linalg.norm(embd_word2))
        return abs(cosine)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f: 
            W, word_voc = pickle.load(f)

        return W, word_voc

    @staticmethod
    def print_n_most_similar(W, n, word, word_voc):
        inv_word_voc = {}
        for key in word_voc.keys():
            value = word_voc[key]
            inv_word_voc[value] = key

        idx_word = word_voc[word]
        nb_words = len(word_voc)
        similarities = []
        for candidate in range(nb_words):
            if candidate != idx_word:
                cosine = W[idx_word].dot(W[candidate])/(np.linalg.norm(W[idx_word])*np.linalg.norm(W[candidate]))
                similarities.append((cosine, inv_word_voc[candidate]))
        print("The {} words that are most similar to {}:".format(n, word))
        sorted_similarities = sorted(similarities, reverse=True)
        for i in range(n):
            print(sorted_similarities[i])
    
    # GENERATE TRAINING SET

    def get_pairs_and_vocs(self, sentences, winSize):
        '''
        Returns (word, context) pairs, dictionnary of (word, index), and dictionnary of (context, index)
        '''
        wc_pairs = []
        word_voc = {}
        word_index = 0
        context_voc = {}
        context_index = 0
        
        for sent in sentences:
            for i in range(len(sent)):
                word = sent[i]
                # Add word + index in word_voc dict
                word_voc, word_index = self.update_voc(word_voc, word, word_index)
                for j in range(max(0,i-winSize), min(i+winSize+1, len(sent))):
                    if i != j:
                        context = sent[j]
                        # Add context + index in context_voc dict
                        context_voc, context_index = self.update_voc(context_voc, context, context_index)
                        wc_pairs.append((word_voc[word], context_voc[context]))
        
        return wc_pairs, word_voc, context_voc


    def get_neg_pairs(self, wc_pairs, word_voc, context_voc, negativeRate):
        word_voc_list = list(word_voc.keys())
        context_voc_list = list(context_voc.keys())
        
        nb_pairs = len(wc_pairs)
        neg_wc_pairs = []

        for wc_pair in wc_pairs:
            word = wc_pair[0]
            for _ in range(negativeRate):
                context_idx = np.random.randint(nb_pairs)
                context = wc_pairs[context_idx][1]
                neg_wc_pairs.append((word,context))

        return neg_wc_pairs

    # PREPROCESS

    def remove_rare_words(self, sentences, minCount):
        fd = freq_dist([word for sent in sentences for word in sent])
        sentences_w_min_count = []
        for sent in sentences:
            sent_w_min_count = []
            for word in sent:
                if fd[word] > minCount:
                    sent_w_min_count.append(word)
            sentences_w_min_count.append(sent_w_min_count)

        return sentences_w_min_count

    '''
    Subsample removes some important words (for example, name of characters) in the corpus. We decided not to use it. 
    Remark: If we had many input corpuses, we would have use a tf-idf to solve this issue
    '''
    #def subsample(self, sentences, threshold):
    #    '''
    #    Returns subsampled sentences, i.e. sentences after having removed words
    #    with proba = 1-sqrt(t/f)
    #    '''
    #    # List of the frequency of the word in the text
    #    fd = freq_dist([word for sent in sentences for word in sent])
    #    
    #    #Create a list of the words to remove according to the subsampling policy
    #    word_removed = []
    #    for word in fd:
    #        p = 1 - sqrt (threshold / fd[word])
    #        if np.random.random() > p:
    #            word_removed.append(word)
    #    
    #    #The output is the original list of sentences from which we take off the word_removed list
    #    sentences_subsampling = sentences
    #    for i in range(len(sentences)):   
    #        sentences_subsampling[i] = [a for a in sentences[i] if a not in word_removed]
    #    
    #return sentences_subsampling

    # COST FUNCTION AND ITS GRADIENT
    def grad_cost_function(self, theta, nEmbed, wc_pairs, neg_wc_pairs, nb_words, nb_contexts):
        '''
        Computes gradient of the cost function described in Goldberg and Levy paper
        '''
        try:
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

        except OverflowError:
            raise DivergentGradientError("The gradient diverged! Decrease the stepsize and/or increase the batchsize.")

        
        return -grad

    # UTILS
    def update_voc(self, voc_dict, word, index):
        # (word: index, count)
        if word not in voc_dict:
            voc_dict[word] = index
            index += 1
        return voc_dict, index

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')

    opts = parser.parse_args()

    if not opts.test:
        sentences = text2sentences(opts.text)
        sg = mySkipGram(sentences)
        sg.train()
        sg.save(opts.model)

    else:
        pairs = loadPairs(opts.text)
        W, word_voc = mySkipGram.load(opts.model)
        #mySkipGram.print_n_most_similar(W, 10, "scone", word_voc)
        
        for a,b,_ in pairs:
            print(a, b, mySkipGram.similarity(a,b,W,word_voc))

