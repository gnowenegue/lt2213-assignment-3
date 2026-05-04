# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: LT2213 Assignment 3
#     language: python
#     name: lt2213-assignment-3
# ---

# %% [markdown]
# # A3: Word Embeddings and Language Modelling
#
# **By Adam Ek, Ricardo Muñoz Sánchez, and Simon Dobnik**

# %% [markdown]
# The lab is an exploration and learning exercise to be done in a group and also in discussion with the teachers and other students.
#
# Before starting, please read the instructions on how to work in groups on Canvas.
#
# Write all your answers and the code in the appropriate boxes below.
#
# In this lab we will explore constructing *static*, *universal* or *general* word embeddings, those that represent meaning of words accross all contexts, as for examplöe implemented in Glove and and word2vec, and then use them with language models which place these embeddings in their syntactic and semantic contexts. We'll also evaluate these systems on tasks such as word similarity and identifying "good" (acceptable) and "bad" (unacceptable) sentences.
#
# **Dependencies**
#
# * Pytorch
#     * Installation instructions: https://pytorch.org/
#     * Tutorials: https://pytorch.org/tutorials/beginner/basics/intro.html
#     * Some useful basic operations: https://jhui.github.io/2018/02/09/PyTorch-Basic-operations
#
#
# **Running the code**
#
# As we are learning about the models, and also what methods work and do not work for our semantic tasks, we are not interested in achievening a state-of-the-art performance but more on details about the implementation and differences in performance under different configurations.
#
# For this reason, you can use a subset of the dataset, for example, the first 5.000-10.000 sentences. On linux or mac you can get these with ``head -n 10000 inputfile > outputfile``.
#
# Using GPUs will make things run faster. You can access the MLT server by using SSH: ``ssh -L 8888:localhost:8888 [your_x_account]@mltgpu.flov.gu.se -p 62266``.
#
# * The ``ssh`` establishes an encrypted shell connection to the server.
# * ``-L 8888:localhost:8888`` maps your local port that Jupyter notebooks on your computer to a port on the server. This allows you to run the code that you see on your computer on the server rather than locally. If you are editing the code directly on the server, you can leave out this step.
# * ``-p 62266`` tells the server to give you access through port 62266.
# * You can also can also connect to the server using VSCode, available for Mac, Linux, and Windows using the details above.
#
# We suggest you to set up a virtual environment on the server, such as ``virtual env`` or ``conda`` within which you then install all the required packages for this lab. When using pytorch on the server, remember to install the GPU-compatible version!
#
# You can also use Google Collab with your free monthly quota for GPUs. However, we strongly recommend that you use the MLT server, though.
#
# **On using generative AI for this assignment:** For this lab, the use of generative AI is permitted as a supporting tool, provided it is done in a responsible and conscious manner and that you state clearly with each question how it was used. However, generative AI must never replace the intellectual work you are expected to carry out. Note that the purpose of this lab is to learn some basic coding of the main neural architectures used in natural language processing. You are responsible for ensuring that such tools are used in a way that supports the development of the skills the module is designed to promote. It is your responsibility to ensure that submitted work is the result of independent intellectual effort.
#
# **Getting help:** We encourage you to use Canvas discussions to post questions and interact with teachers and also each other. Provide youseful tips, but of course do not reveal the exact answer across the groups as each group should should work out their own solutions. Remember that in most cases there is also not a single correct answer and implementations may differ.

# %%
# %pip install -r packages.txt

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# If you're using GPUs, replace "cpu" with "cuda:n" where n is the index of the GPU
if torch.cuda.is_available():
    device = torch.device('cuda:0')  # use the first GPU
else:
    device = torch.device('cpu')

# %% [markdown]
# # Word2Vec embeddings
#
# We will construct a word2vec model which will give us *static* or *genral* word embeddings, that is embeddings for words regardless of the their sentantial or situational contexts, that therefore cover all word senses.
#
# Then we will evaluate the embeddings in a word similarity task.

# %% [markdown]
# ## Formatting data
#

# %% [markdown]
# First we need to load some data from data/wiki-corpus.50000.txt. The file contains 50.000 sentences randomly selected from the complete wikipedia. Each line in the file contains one sentence. The sentences are whitespace tokenised.
#
# Create a dataset suitable for word2vec. To do this, we define some ```window_size``` and then iterate over all sentences in the dataset, putting the target word in one column and the context words in another (separate the columns with ```tab```). ```window_size=n``` means that we select ```n/2``` tokens to the right and left of the center word.
#
# For example, the sentece "this is a lab and exercise" with ```window size = 4``` will be converted to 6 (target, context) pairs:
#
# ```
# target      context
# ----------------------------
# this        is, a
# is          this, a, lab
# a           this, is, lab
# lab         is, a, and, exercise
# and         a, lab, exercise
# exercise    lab, and
# ```
#
# which will be our training examples for the word2vec model.
#
# [3 marks]

# %%
from collections import Counter
data_path = 'data/wiki-corpus.50000.txt'
WINDOW_SIZE = _
def corpus_reader(data_path, window_size=4, min_freq=4):
    all_data = []
    vocabulary = set(['<pad>'])
    counter = Counter()
    with open(data_path) as f:
        # go over the lines (sentences in the files)
        for line in f:
            line = line.strip()
        # split sentences into tokens
            tokens = line.split()
        # save all indiviual words to the vocabulary
            vocabulary.update(tokens)
            counter.update(tokens)
        # extract all (center word, context) with `window_size=4`, pairs from the sentence
            for i, center_word in enumerate(tokens):
                context = []
                #left
                for j in range(i-(window_size//2),i):
                    if j>=0:
                        context.append(tokens[j])
                    else:
                        context.append("<pad>")
                #Right
                for j in range(i+1,i+(window_size//2)+1):
                    if j<len(tokens):
                        context.append(tokens[j])
                    else:
                        context.append("<pad>")

        # save (center word, context) pairs into a dataset
                all_data.append((center_word, context))

    # filter out words which does not occur often
    temp = set()
    for word in vocabulary:
        if counter[word] >= min_freq:
            temp.add(word)
    vocabulary = temp

    # create a mapping from words to integers.
    # each word should have an unique integer mapped to it.
    # use a dictionary for this.
    word_to_idx = {}
    for idx, word in enumerate(vocabulary):
        word_to_idx[word] = idx
    return all_data, word_to_idx

all_data, word_to_idx = corpus_reader(data_path)

# %%
# # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # EUGENE'S ATTEMPT # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # #

from collections import Counter


data_path = './data/wiki-corpus.50000.txt'
WINDOW_SIZE = 4


def corpus_reader(data_path, window_size=4, min_freq=4):
    all_data = []
    vocabulary = set(['<pad>'])
    counter = Counter()

    with open(data_path) as f:
        # go over the lines (sentences in the files)
        for line in f:
            line = line.strip()
        # split sentences into tokens
            tokens = line.split()
        # save all indiviual words to the vocabulary
            vocabulary.update(tokens)
            counter.update(tokens)
        # extract all (center word, context) with `window_size=4`, pairs from the sentence
            half_window = window_size // 2

            for i, center_word in enumerate(tokens):
                # print(center_word)
                context_words = []

                # left context
                for j in range(i - half_window, i):
                    if j >= 0:
                        context_words.append(tokens[j])
                    # else:
                    #     context_words.append('<pad>')

                # right context
                for j in range(i + 1, i + half_window + 1):
                    if j < len(tokens):
                        context_words.append(tokens[j])
                    # else:
                    #     context_words.append('<pad>')
        # save (center word, context) pairs into a dataset
                all_data.append((center_word, context_words))

    # filter out words which does not occur often
    filtered_vocabulary = set(['<pad>'])

    for word in vocabulary:
        if counter[word] >= min_freq:
            filtered_vocabulary.add(word)

    vocabulary = filtered_vocabulary

    # create a mapping from words to integers.
    # each word should have an unique integer mapped to it.
    # use a dictionary for this.
    word_to_idx = {
        '<pad>': 0, # padding
        '<unk>': 1  # unknown words
    }

    current_idx = 2  # start indexing from 2 since 0 and 1 are reserved for <pad> and <unk>

    for word in vocabulary:
        if word not in word_to_idx:
            word_to_idx[word] = current_idx
            current_idx += 1

    return all_data, word_to_idx


all_data, word_to_idx = corpus_reader(data_path, WINDOW_SIZE)

# %%
#testing.
print(len(all_data))
print(all_data[0])
print(len(word_to_idx))

# %% [markdown]
# We sampled 50.000 senteces randomly from the *entire* wikipedia for our training data. Give some reasons why this is good, and why it might be bad. (*note*: We'll have a few questions like these, one or two reasons for and against is sufficient)
#
# [2 marks]

# %%

# %% [markdown]
# ## Loading data
#
# We need to create a dataloader which will read the data and generate batches of examples from the dataset. A batch is a set of ```n``` examples from the data.
#
# The recipe for a dataloader is as follows:
#
# * Select n examples from the dataset
# * (a) Translate each example into integers using `word_to_idx`
# * (b) Transform the translated examples to pytorch tensors
# * (c) Return the batch
# * Select n new examples from the dataset
# * ... repeat steps (a-c)
#
# The dataloader should stop when it has read the entire dataset.
#
# This can be done either by first computing all the batches in the dataset and returning them as a list which you can then iterate over, or as an generator that returns each batch after it has been created.
#
# [4 marks]

# %%
from collections import namedtuple
Batch = namedtuple('Batch', ['target_word', 'context'])

def batcher(dataset, word_to_idx, batch_size=8):
    # iterate over the dataset
    for i in range(0, len(dataset), batch_size):
    # select a batch of size `batch_size`
        batch_data = dataset[i : i + batch_size]
        batch_targets = []
        batch_contexts = []
    # translate batch to integers using `word_to_idx`
        for center_word, context_words in batch_data:
            target_idx = word_to_idx.get(center_word, 0)
            context_idx = [word_to_idx.get(w, 0) for w in context_words]

            batch_targets.append(target_idx)
            batch_contexts.append(context_idx)
    # add padding to the context
        max_len = max(len(c) for c in batch_contexts)
        pad_id = word_to_idx.get('<pad>', 0)
        for c in batch_contexts:
            while len(c) < max_len:
                c.append(pad_id)
    # transform the batch to a pytorch tensor
        target_tensor = torch.tensor(batch_targets, dtype=torch.long)
        context_tensor = torch.tensor(batch_contexts, dtype=torch.long)
    # return the dataset of batches/indiviual batches
        batch = Batch(target_tensor, context_tensor)
        yield batch


# %%
# # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # EUGENE'S ATTEMPT # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # #

from collections import namedtuple
Batch = namedtuple('Batch', ['target_word', 'context'])


def batcher(dataset, word_to_idx, batch_size=8):
    # iterate over the dataset
    for i in range(0, len(dataset), batch_size):
        # select a batch of size `batch_size`
        batch_data = dataset[i: i + batch_size]
        batch_targets = []
        batch_contexts = []

    # translate batch to integers using `word_to_idx`
        for center_word, context_words in batch_data:
            # default to <unk> index if word not found
            target_idx = word_to_idx.get(center_word, 1)
            context_idx = [word_to_idx.get(w, 1) for w in context_words]

            batch_targets.append(target_idx)
            batch_contexts.append(context_idx)

    # add padding to the context
        max_len = max(len(c) for c in batch_contexts)

        for context in batch_contexts:
            while len(context) < max_len:
                context.append(0)  # pad with <pad> index

    # transform the batch to a pytorch tensor
        target_tensor = torch.tensor(batch_targets, dtype=torch.long)
        context_tensor = torch.tensor(batch_contexts, dtype=torch.long)

    # return the dataset of batches/indiviual batches
    batch = Batch(target_word=target_tensor, context=context_tensor)
    yield batch


# %% [markdown]
# We lower-cased all tokens above. Give some reasons why this is a good idea, and why it may be harmful to our embeddings.
#
# [2 marks]

# %%

# %% [markdown]
# ## Word embeddings model

# %% [markdown]
# We will implement the CBOW model for constructing word embedding models.

# %%
import torch.optim as optim


# %% [markdown]
# In the CBOW model we try to predict the center word based on the context. We take as input ```n``` context words, encode them as vectors, and combine them by summation. This will give us one embedding. We then use this embedding to predict *what* word in our vocabuary is the most likely center or target word.
#
# Implement this model by completing the code below.
#
# [7 marks]

# %%
class CBOWModel(nn.Module):
    def __init__(self, ...):
        super(CBOWModel, self).__init__()
        # where the embeddings of words are stored
        # each word in the vocabulary should have one embedding assigned to it
        self.embeddings = ...
        # a transformation that predicts a word from the vocabulary
        self.prediction = ...

    def forward(self, context):
        # translate a batch to embeddings
        embedded_context = ...
        # reduce dimensions of the embeddings
        projection = ...
        # predict the target word from the vocabulary
        predictions = ...

        return predictions

    def projection_function(self, xs):
        """
        This function will take as input a tensor of size (B, S, D)
        where B is the batch_size, S the window size, and D the dimensionality of embeddings
        this function should compute the sum over the embedding dimensions of the input,
        that is, we transform (B, S, D) to (B, 1, D) or (B, D)
        """
        xs_sum = ...
        return xs_sum


# %% [markdown]
# The next step is to train a model. First we define what (hyper)parameters we will use, i.e. settings that affect how the model will be trained. You can change these and see what happens with training, for example when *developing* your model you can use a batch size of 2 and a very low dimensionality (say 10) to speed things up. For training your final target model, use batch sizes of [8,16,32,64], and embedding dimensionalities [128,256].

# %%
word_embeddings_hyperparameters = {'epochs':3,
                                   'batch_size':16,
                                   'learning_rate':0.001,
                                   'embedding_dim':128}

# %% [markdown]
# Train your model. Iterate over the dataset, get outputs from your model, calculate loss and backpropagate.
#
# Frequently, Negative Log Likelihood (https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html) loss is used to train a Word2Vec model. However, in this lab we will take a *training* shortcut and use Cross Entropy Loss (https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html), instead which combines ```log_softmax``` and ```NLLLoss```. Your model should thus output a *score* for each word in the vocabulary. The ```CrossEntropyLoss``` will assign probabilities and calculate the negative log likelihood loss.
#
# [3 marks]

# %%
# load data
dataset, vocab = get_data(...)

# build model and construct loss/optimizer
cbow_model = CBOWModel(len(vocab), word_embeddings_hyperparameters['embedding_dim'])
cbow_model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(cbow_model.parameters(), lr=word_embeddings_hyperparameters['lr'])

# start training loop
total_loss = 0
for epoch in range(word_embeddings_hyperparameters['epochs']):
    for i, batch in enumerate(dataset):

        context = batch.context
        target_word = batch.target_word

        # send your batch of sentences to the model
        output = cbow_model(context)

        # compute the loss, you'll need to reshape the input
        # you can read more about this is the documentation for
        # CrossEntropyLoss
        loss = loss_fn(...)
        total_loss += loss.item()

        # print average loss for the epoch
        print(total_loss/(i+1), end='\r')

        # compute gradients
        ...

        # update parameters
        ...

        # reset gradients
        ...
    print()


# %% [markdown]
# ## Evaluating the model

# %% [markdown]
# We will evaluate our models on a dataset of word similarities, WordSim353 (http://alfonseca.org/eng/research/wordsim353.html , also avalable in the data folder). We need to read the dataset and translate it to integers. To do this we will reuse the ```Field``` that records word indexes (the second output of ```get_data()```) and use it to parse the file.
#
# The wordsim data has the following structure:
#
# ```
# word1 word2 score
# ...
# ```
#
# The ```Field``` we get from ```read_data()``` has two built-in functions, ```stoi``` which maps a string to an integer and ```itos``` which maps an integer to a string.
#
# Your datareader should do the following:
#
# ```
# for line in file:
#     word1, word2, score = file.split()
#     # encode word1 and word2 as integers
#     word1_idx = vocab.vocab.stoi[word1]
#     word2_idx = vocab.vocab.stoi[word2]
# ```
#
# When we have the integers for ```word_1``` and ```word2``` will compute the similarity between their word embeddings using the *cosine simlarity* score. We obtain the embeddings by querying the embedding layer of the model.
#
# We calculate the cosine similarity for each word pair in the dataset, then compute the Pearson correlation coefficient between the similarities predicted by our models and the human-reported scores from the dataset.
#
# [4 marks]

# %%
def read_wordsim(path, vocab, embeddings):
    dataset_sims = []
    model_sims = []
    with open(path) as f:
        for line in f:
            word1, word2, score = f.split()

            score = float(score)
            dataset_sims.append(score)

            # get the index for the word
            word1_idx = ...
            word2_idx = ...

            # get the embedding of the word
            word1_emb = ...
            word2_emb = ...

            # compute cosine similarity, we'll use the version included in pytorch functional
            # https://pytorch.org/docs/master/generated/torch.nn.functional.cosine_similarity.html
            cosine_similarity = F.cosine_similarity(...)

            model_sims.append(cosine_similarity.item())

    return dataset_sims, model_sims

path = 'wordsim_similarity_goldstandard.txt'
data, model = read_wordsim(...)
pearson_correlation = np.corrcoef(data, model)

# the non-diagonals give the pearson correlation,
print(pearson_correlation)

# %% [markdown]
# Do you think the model performs well or not? Why?
#
# [3 marks]

# %%

# %% [markdown]
# Select the 10 best and 10 worst performing word pairs. Can you see any patterns that explain why *these* are the best and worst word pairs?
#
# [3 marks]

# %%

# %% [markdown]
# Suggest some ways of improving the model for the task in WordSim353.
#
# [3 marks]

# %%

# %% [markdown]
# Sentiment analysis is a (downstream, i.e. a follow-up task) where a model is like this might be used for training. It involves determining whether a sentence is positive or negative.
#
# Give some examples why a sentiment analysis model would benefit from our embeddnings and some examples why our embeddings would not work well for the sentiment analysis model.
#
# [3 marks]

# %%

# %% [markdown]
# # Language modeling

# %% [markdown]
# We will also build a simple LSTM language model which can be seen as a downstream task. Construct a model which takes a sentence as an input, one word at a time, and predicts the next word for each word in the sentence. For this you'll use the ```LSTM``` class provided by PyTorch (https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html). You can read more about the LSTMs here: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
#
# Use the same dataset (wiki-corpus.50000.txt) as before.
#
# The procedure is similiar to before. We first encode the words as distributed representations and then pass these to the LSTM and for each output we predict the next word.
#
# To produce inputs and outputs for training we need to conver the sentence representing a sentence to a tensor. As we want to predict the *next* word, we want a setup where `w_n` is the index of a word in the sentence, `x` is the input words, and `y` is the target words):
#
# $x = [w_0, w_1, w_2, w_3, w_4]$
#
# $y = [w_1, w_2, w_3, w_4, w_5]$
#
# That is, to create a target word we need to shift the index `n` of the input by `+1`, as this gives us the next word.
#
# For this we will build a new dataloader that will read a file containing one sentence per line, with words separated by whitespace.
#
# ```
# word_1, ..., word_n
# word_1, ..., word_k
# ...
# ```
#
# In the dataloader each sentence should begin with a ```<start>``` token and end with a ```<end>``` token to mark the beginning and end of sentences. The remeining steps are the same as before: you read the dataset and output an iterator over the dataset, a vocabulary, and a mapping from words to indices.
#
# Implement a dataloader, a language model and a training loop (the latter will be nearly identical to the one for word2vec).
#
# [12 marks]

# %%
# you can change these numbers to suit your needs as before
lm_hyperparameters = {'epochs':3,
                      'batch_size':16,
                      'learning_rate':0.001,
                      'embedding_dim':128,
                      'output_dim':128}

# %%
data_path = 'wiki-corpus.txt'
def get_data():
    # your code here, roughly the same as for the word2vec dataloader


# %%
class LM_withLSTM(nn.Module):
    def __init__(...):
        super(LM_withLSTM, self).__init__()
        self.embeddings = ...
        self.LSTM = nn.LSTM(self, input_size=..., hidden_size=...)
        self.predict_word = ...

    def forward(self, seq):
        # extract embeddings for the sentence
        embedded_seq = ...
        # compute contextual representations
        timestep_reprentation, *_ = self.LSTM(embedded_seq)
        # predict a token from the vocabulary at each timestep
        predicted_words = ...

        return predicted_words


# %%
# load data
dataset, vocab = get_data(...)

# build model and construct loss/optimizer
lm_model = LM_withLSTM(len(vocab),
                       lm_hyperparameters['embedding_dim'],
                       lm_hyperparameters['output_dim'])
lm_model.to(device)

loss_fn = CrossEntropyLoss()
optimizer = optim.Adam(cbow_model.parameters(), lr=lm_hyperparameters['lr'])

# start training loop
total_loss = 0
for epoch in range(lm_hyperparameters['epochs']):
    for i, batch in enumerate(dataset):

        # the strucure for each BATCH is:
        # <start>, w0, ..., wn, <end>
        sentence = batch.sentence

        # when training the model, at each input we predict the *NEXT* token
        # consequently there is nothing to predict when we give the model
        # <end> as input.
        # thus, we do not want to give <end> as input to the model, select
        # from each batch all tokens except the last.
        # tip: use pytorch indexing/slicing (same as numpy)
        # (https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html#operations-on-tensors)
        # (https://jhui.github.io/2018/02/09/PyTorch-Basic-operations/)
        input_sentence = ...

        # send your batch of sentences to the model
        output = lm_model(input_sentence)

        # for each output, the model predict the NEXT token, so we have to reshape
        # our dataset again. On timestep t, we evaluate on token t+1. That is,
        # we never predict the <start> token ;) so this time, we select all but the first
        # token from sentences (that is, all the tokens that we predict)
        gold_data = ...

        # the shape of the output and sentence variable need to be changed,
        # for the loss function. Details are in the documentation.
        # You can use .view(...,...) to reshape the tensors
        loss = loss_fn(...)
        total_loss += loss.item()

        # print average loss for the epoch
        print(total_loss/(i+1), end='\r')

        # compute gradients
        ...

        # update parameters
        ...

        # reset gradients
        ...
    print()

# %% [markdown]
# ## Evaluating your language model
#
# We will evaluate our language model using the BLiMP dataset (https://github.com/alexwarstadt/blimp) which contains sets of linguistic minimal pairs exemplifying various syntactic and semantic phenomena. One of these are *existential quantifiers* (link: https://github.com/alexwarstadt/blimp/blob/master/data/existential_there_quantifiers_1.jsonl). This dataset, as the name suggests, is intended to investigate whether language models assign higher probability to *correct* (i.e. *acceptable* if you are a linguist!) usages of there-quantifiers.
#
# An example entry from the dataset:
#
# ```
# {"sentence_good": "There was a documentary about music irritating Allison.", "sentence_bad": "There was each documentary about music irritating Allison.", "field": "semantics", "linguistics_term": "quantifiers", "UID": "existential_there_quantifiers_1", "simple_LM_method": true, "one_prefix_method": false, "two_prefix_method": false, "lexically_identical": false, "pairID": "0"}
# ```
#
# Download the dataset and build a datareader similar to the one for word2vec. The relevant dataset structure are the following: (You  can ignore the other keys in this assignment.)
#
# ```
# good_sentence_1, bad_sentence_1
# ...
# ```
#
# Compare the probability assigned to a good sentence with to the probability assigned to a bad sentence. To compute the probability for a sentence as a whole we consider the product of the probabilities assigned to the *target* tokens. (Here we could also choose other methods.) Remember, at timestep ```t``` we're predicting what token comes *next* e.g. ```t+1``` Hence, the procedure is similar to training.
#
# Here is some rough pseudo code that you can base your answer:
#
# ```
# accuracy = []
# for good_sentence, bad_sentence in dataset:
#     gs_lm_output = LanguageModel(good_sentence)
#     gs_token_probabilities = softmax(gs_lm_output)
#     gs_sentence_probability = product(gs_token_probabilities[GOLD_TOKENS])
#
#     bs_lm_output = LanguageModel(bad_sentence)
#     bs_token_probabilities = softmax(bs_lm_output)
#     bs_sentence_probability = product(bs_token_probabilities[GOLD_TOKENS])
#
#     # int(True) = 1 and int(False) = 0
#     is_correct = int(gs_sentence_probability > bs_sentence_probability)
#     accuracy.append(is_correct)
#
# print(numpy.mean(accuracy))
#
# ```
#
# [6 marks]

# %%
# your code goes here
import json

def evaluate_model(path, vocab, model):

    accuracy = []
    with open(path) as f:
        # iterate over one pair of sentences at a time
        for line in f:
            # load the data
            data = json.loads(line)
            good_s = data['sentence_good']
            bad_s = data['sentence_bad']

            # the data is tokenized as whitespace
            tok_good_s = ...
            tok_bad_s = ...

            # encode your words as integers using the vocab from the dataloader, size is (S)
            # we use unsqueeze to create the batch dimension
            # in this case our input is only ONE batch, so the size of the tensor becomes:
            # (S) -> (1, S) as the model expects batches
            enc_good_s = torch.tensor([_ for x in tok_good_s], device=device).unsqueeze(0)
            enc_bad_s = torch.tensor([_ for x in tok_bad_s], device=device).unsqueeze(0)

            # pass your encoded sentences to the model and predict the next tokens
            good_s = LM_withLSTM(enc_good_s)
            bad_s = LM_withLSTM(enc_bad_s)

            # get probabilities with softmax
            gs_probs = F.softmax(...)
            bs_probs = F.softmax(...)

            # select the probability of the gold tokens
            gs_sent_prob = find_token_probs(gs_probs, enc_good_s)
            bs_sent_prob = find_token_probs(bs_probs, enc_bad_s)

            accuracy.append(int(gs_sent_prob>bs_sent_prob))

    return accuracy

def find_token_probs(model_probs, encoded_sentece):
    probs = []

    # iterate over the tokens in your encoded sentence
    for token, gold_token in enumerate(encoded_sentece):
        # select the probability of the gold tokens and save
        # hint: pytorch indexing is helpful here ;)
        prob = ...
        probs.append(prob)
    sentence_prob = ...
    return sentence_prob

path = 'existential_there_quantifiers_1.jsonl'
accuracy = evaluate_model(path, ..., ...)

print('Final accuracy:')
print(np.round(np.mean(accuracy), 3))


# %% [markdown]
# ## Analysis and discussion

# %% [markdown]
# If our model achieves to predict the correct label in 55% of cases, is this a good performance? Suggest a *baseline* (i.e. "model" based on some simple method, typcially one that does not involve training and that we aim to beat) that we can compare the model against.
#
# [3 marks]

# %%

# %% [markdown]
# Suggest some improvements you could make to your language model.
#
# [3 marks]

# %%

# %% [markdown]
# Suggest some other metrics we can use to evaluate our system
#
# [2 marks]

# %%

# %% [markdown]
# # Literature
#
#
# [1] Y. Bengio, R. Ducharme, P. Vincent, and C. Janvin. A neural probabilistic language model. (Links to an external site.) Journal of Machine Learning Research, 3(6):1137–1155, 2003. (Sections 3 and 4 are less relevant today and hence you can glance through them quickly. Instead, look at the Mikolov papers where they describe training word embeddings with the current neural network architectures.)
#
# [2] T. Mikolov, K. Chen, G. Corrado, and J. Dean. Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781, 2013.
#
# [3] T. Mikolov, I. Sutskever, K. Chen, G. S. Corrado, and J. Dean. Distributed representations of words and phrases and their compositionality. In Advances in neural information processing systems, pages 3111–3119, 2013.
#
#

# %% [markdown]
# # Your reflections on this lab
#
# Write below your general thoughts, experiences, or reflections on how you worked on this lab.

# %%

# %% [markdown]
# ## Statement of contribution
#
# Briefly state how many times you have met for discussions, who was present, to what degree each member contributed to the discussion and the final answers you are submitting.

# %%

# %% [markdown]
# ## Marks
#
# The assignment is marked on a 7-level scale where 4 is sufficient to complete the assignment; 5 is good solid work; 6 is excellent work, covers most of the assignment; and 7: creative work.
#
# This assignment has a total of 60 marks. These translate to grades as follows: 1 = 17% 2 = 34%, 3 = 50%, 4 = 67%, 5 = 75%, 6 = 84%, 7 = 92% where %s are interpreted as lower bounds to achieve that grade.
