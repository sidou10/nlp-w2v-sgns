# word2vec (skip-gram with negative sampling version)

This version of word2vec implements a well-choosen log-likelihood function that is maximised using stochastic gradient ascent.
- Input: path to text file
- Output: word embeddings of each word

## Preprocessing
- The **text2sentences** function transforms a raw text in input file to tokenized sentences. The sentences are delimited by ".", "?" or "!"
- **Rare words**, i.e. words appearing less than minCount in the initial text, are removed. minCount is a parameter of the mySkipGram class and can be modified.
- **Subsampling** (removing with high probability very frequent words) can be activated by uncommenting the code. It is known as a good practice to improve the algorithm performance. However, it may remove some important words such as characters in a story (achilles in odyssey).

## Train mode
Example of command line to execute
```
python skipGram.py --text data/odyssey.txt --model odyssey_model
```
The command uses the odyssey.txt as training set and saves the word embeddings in odyssey_model file.

Example of standard output for the previous command
```
Parameters: nEmbed=100, negativeRate=5, winSize=3, minCount=5
Generating training set and vocabularies... ok!
Generating negative samples... ok!
TRAINING: #epochs: 5, step size: 0.01, batch size: 512
Epoch 1/5
(100.34s)
Epoch 2/5
(97.96s)
Epoch 3/5
(97.85s)
Epoch 4/5
(98.24s)
Epoch 5/5
(101.11s)
```
In some cases, depending on the training set, the gradient may diverge. In this case, a DivergentGradientError is raised. The stepsize should be reduced or the batch size increased. This is to be done directly in the .py file.


## Test mode
Example of command line to execute
```
python skipGram.py --text data/test_odyssey.csv --model odyssey_model --test
```

This command uses the previous created model and computes the cosine similarity between pairs of words

Example of standard output for the previous command
```
achilles apollo 0.969467657893
brother father 0.949372995447
achilles brother 0.813859138363
hector achilles 0.954174722323
Unknown word 'blue' mapped to default embedding (0.1)
hector blue 0.278300089587
```

