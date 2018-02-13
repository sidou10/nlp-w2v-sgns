# word2vec (skip-gram with negative sampling version)

This version of word2vec implements a well-choosen log-likelihood function that is maximised using stochastic gradient ascent.
- Input: path to text file
- Output: word embeddings of each word

## Preprocessing
- The **text2sentences** function transforms a raw text in input file to tokenized sentences. The sentences are delimited by ".", "?" or "!"
- **Rare words**, i.e. words appearing less than minCount in the initial text, are removed. minCount is a parameter of the mySkipGram class and can be modified.
- **Subsampling** (removing with high probability very frequent words) can be activated by uncommenting the code. It is known as a good practice to improve the algorithm performance. However, it may remove some important words such as characters in a story (achilles in odyssey).

## Train mode
Example of command line to execute:
```
python skipGram.py --text data/odyssey.txt --model odyssey_model
```
The command uses the odyssey.txt as training set and saves the word embeddings in odyssey_model file.

Example of standard output for the previous command:
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
Note that an epoch lasts approximately 100 seconds for the odyssey text that contains 152601 words (and 4906 sentences).

In some cases, depending on the training set, the gradient may diverge. In this case, a DivergentGradientError is raised. The stepsize should be reduced or the batch size increased. This is to be done directly in the .py file.


## Test mode
Example of command line to execute:
```
python skipGram.py --text data/test_odyssey.csv --model odyssey_model --test
```

This command uses the previous created model and computes the cosine similarity between pairs of words.

Example of standard output for the previous command:
```
achilles apollo 0.969467657893
brother father 0.949372995447
achilles brother 0.813859138363
hector achilles 0.954174722323
Unknown word 'blue' mapped to default embedding (0.1)
hector blue 0.278300089587
```

## Interesting results
Training on oddyssey.txt and printing the 10 words the most similar to "achilles"
```
(0.96946765789313449, 'apollo')
(0.96525814365212759, 'menelaus')
(0.95417472232343603, 'hector')
(0.95381790691523705, 'minerva')
(0.93975999195393534, 'patroclus')
(0.91096764302159794, 'juno')
(0.90590132336247331, 'him')
(0.90334624606369707, 'venus')
(0.87159501343202117, 'aeneas')
(0.87143968241241565, 'idomeneus')
```

Training on sentences.txt and printing the 10 most similar to "blue"
```
(0.9563631433212999, 'green')
(0.9526111037594357, 'red')
(0.93553123647789915, 'yellow')
(0.92785787371694561, 'white')
(0.92255178953816452, 'blackhaired')
(0.92219070113099944, 'purple')
(0.920377432232246, 'leather')
(0.91389369124582287, 'black')
(0.91328527016416539, 'caucasian')
(0.91262585867689616, 'denim')
```

Training on sentences.txt and printing the 10 most similar to "man"
```
(0.92840613535955896, 'woman')
(0.88753333292909398, 'guy')
(0.88250537367382531, 'girl')
(0.87909185804069612, 'child')
(0.86959913231018127, 'jacket')
(0.86877235522624241, 'women')
(0.86385576996025026, 'lady')
(0.85740224939248577, 'collar')
(0.84601751259737101, 'female')
(0.83840653298179013, 'outfit')
```

Training on sentences.txt and printing the 10 most similar to "
## Interesting results
