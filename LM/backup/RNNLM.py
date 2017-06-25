# script for running iterations of training

import numpy as np
import os
from random import shuffle
import re
import tensorflow as tf
import sys
import urllib.request
import zipfile
import lxml.etree
import json
import math

# initialize the dimension variables
DIM_VOCAB = 0
DIM_WordEmbedding = 100
DIM_Label = 8
DIM_Hidden = 50
DIM_RNN = 100
BATCH_SIZE = 50
EPOCH = 10
PADDING = 'pad0'
UNKNOWN = 'UKN'
STARTWORD = 'STARTWORD'
STOPWORD = 'STOPWORD'
WORDCOUNTS = [] # train/valid/test

# Download the dataset if it's not already there: this may take a minute as it is 75MB
if not os.path.isfile('ted_en-20160408.zip'):
    urllib.request.urlretrieve("https://wit3.fbk.eu/get.php?path=XML_releases/xml/ted_en-20160408.zip&filename=ted_en-20160408.zip", filename="ted_en-20160408.zip")
    
# For now, we're only interested in the subtitle text, so let's extract that from the XML:
with zipfile.ZipFile('ted_en-20160408.zip', 'r') as z:
    doc = lxml.etree.parse(z.open('ted_en-20160408.xml', 'r'))
input_text = doc.xpath('//content/text()')
    
for i in range(len(input_text)):
    # for each input_text, remove the parenthesise
    input_text[i] = re.sub(r'\([^)]*\)', '', input_text[i])
    # the input_text contains a lot of new lines
    temp = []
    input_text[i] = ''.join(input_text[i].split('\n'))
        
temp = []
count = 0
for block in input_text:
    m = re.match(r'^(?:(?P<precolon>[^:]{,20}):)?(?P<postcolon>.*)$', block)
    temp.append([sent for sent in m.groupdict()['postcolon'].split('.') if sent])
input_text = temp

for i in range(len(input_text)):
    for j in range(len(input_text[i])):
        tokens = re.sub(r"[^a-z0-9]+", " ", input_text[i][j].lower()).split()
        input_text[i][j] = tokens

from prepare_sentences import Preparer
P = Preparer(DIM_RNN, input_text)
P.addStartStopWords(STARTWORD, STOPWORD)
docLengths = P.cutDocs() # docLengths are the list of unpadded lengths of each sentence
WORDCOUNTS = P.getTotalWordCount()
print (WORDCOUNTS)

# IMPORTANT - format of data:
# data[i][0]: document (list of fixed-size sentences (list of words))
# data[i][1]: list of (sentence lengths for each sentence) for each doc
text_labels = [[input_text[i], docLengths[i]] for i in range(len(input_text))]
training_data = text_labels[:1585]
validation_data = text_labels[1585:1835]
testing_data = text_labels[1835:]


from gensim.models import Word2Vec
# import the script which trains and saves word2Vec model
import trainWord2Vec 
# loading W2V model
def getWord2Vec():
    if os.path.isfile('word2Vec'):
        model = Word2Vec.load('word2Vec')
    else:
        # execute python script to train a word2Vec model
        sentences = []
        for tl in training_data:
            text = tl[0]
            sentences += text
        trainWord2Vec.train(sentences)
        model = Word2Vec.load('word2Vec')
    return model
        
# Mword as the initialization word embedding
Mword = None
embeddingName = 'Word2Vec'
def getWordEmbedding(embeddingName):
    if (embeddingName == 'Word2Vec'):
        model = getWord2Vec()
    elif (embeddingName == 'Glove'):
        if (os.path.isfile('glove.6B.zip')):
            # TODO: unzip and load the glove model
            model = None
    else:
        # start with a random model
        model = None
    return model
w2v_model = getWordEmbedding(embeddingName)

Mword = w2v_model.syn0
Mword = np.concatenate((Mword, [[0.0]*DIM_WordEmbedding]), axis = 0)
Mword = np.concatenate((Mword, [np.random.uniform(-1.0, 1.0, size = [DIM_WordEmbedding])]), axis = 0)
Mword = np.concatenate((Mword, [np.random.uniform(-1.0, 1.0, size = [DIM_WordEmbedding])]), axis = 0)
Mword = np.concatenate((Mword, [np.random.uniform(-1.0, 1.0, size = [DIM_WordEmbedding])]), axis = 0)
print (Mword.shape)
word_index = w2v_model.index2word
word_index.append(PADDING)
word_index.append(UNKNOWN)
word_index.append(STARTWORD)
word_index.append(STOPWORD)
if (embeddingName == 'Word2Vec'):
    DIM_VOCAB, DIM_WordEmbedding = Mword.shape
    
import tensorflow as tf
# construct a word dictionary, word - index
word_dic = {}
if not os.path.isfile('wordDic'):
    word_dic = {}
    for w in word_index:
        word_dic[w] = len(word_dic)
    # store the dictionary into json
    json.dump(word_dic, open('wordDic', 'w'))
else:
    #load the dictionary from json
    word_dic = json.load(open('wordDic', 'r'))
print ('dictionary loaded')

# prepare the documents as a list of integers
def doc2Int(data):
    # note that the texts are contained in data[0]
    for i in range(len(data)):
        # parse the doc
        # data[i][0] is the list/doc of list/sentence of words
        nums = []
        for j in range(len(data[i][0])):
            # data[i][0][j] is the j'th sentence, sentence is a list of ints
            sentence = []
            for w in data[i][0][j]:
                if w in word_dic:
                    sentence.append(word_dic[w])
                else:
                    sentence.append(word_dic[UNKNOWN])
            nums.append(sentence)
        data[i][0] = nums

doc2Int(training_data)
doc2Int(validation_data)
doc2Int(testing_data)
    

###################
### RNN NETWORK ###
###################
import copy

def getMask(lens):
    # lens: the list of lengths of the sequence (max = DIM_RNN)
    # mask dim: [BATCH_SIZE, DIM_RNN]
    mask = []
    for i in range(BATCH_SIZE):
        temp = [1.0 if j<lens[i] else 0.0 for j in range(DIM_RNN)]
        mask.append(temp)
    return np.array(mask)
# For batching
def getSentenceBatch(batch, ns):
    # batch is the list of documents
    # we need to get the ns'th sentence of each batch, return dim: [BS*DIM_RNN]
    return [doc[ns] for doc in batch]    
def getLabelsBatch(batch, ns, sparse = True):
    # batch is the lsit of documents
    # we need to get the ns'th sentence minus the first word, adding the next word
    sen1batch = copy.deepcopy([doc[ns] for doc in batch])
    #sen1batch = copy.deepcopy(sen1batch)
    max_num_sentence = len(batch[0])
    last_words = None

    if (ns<max_num_sentence-1):
        last_words = [doc[ns+1][0] for doc in batch]
        # append these words to the sen1batch
    else:
        last_words = [doc[ns][-1] for doc in batch] 
  
            
    [sen1batch[i].append(last_words[i]) for i in range(len(last_words))]
    for i in range(len(batch)):
        sen1batch[i] = sen1batch[i][1:]
    #if 'sparse': return the ints - for training data
    if (sparse):
        return sen1batch
    #if not sparse: extend the integer to one hot vectors - for testing/validation data
    for i in range(len(batch)):
        #sen1batch[i] is the list of ints for the ith sentence
        temp = []
        for j in range(len(sen1batch[i])):
            zeroVec = [0]*DIM_VOCAB
            zeroVec[sen1batch[i][j]] = 1
            temp.append(zeroVec)
        sen1batch[i] = temp
    return sen1batch
    
    
def getSentenceLength(lens_batch, ns):
    # batch is the list of documents, each containing one length for each sentence
    # we need a list of lengths for each sentence in the batch, return dim: [BS]
    return [doc[ns] for doc in lens_batch] 

from shuffle import shuffle as SF

with tf.variable_scope('all', reuse = None):
    # INPUT LAYER
    train_x = tf.placeholder(tf.int32, [BATCH_SIZE, DIM_RNN], name= 'train_x') #integer list input
    sparse_train_y = tf.placeholder(tf.int32, [BATCH_SIZE, DIM_RNN])
    # EMBEDDING LAYER
    E = tf.Variable(tf.constant(0.0, shape = [DIM_VOCAB, DIM_WordEmbedding]), trainable=True, name = 'E')
    embedding_placeholder = tf.placeholder(tf.float32, [DIM_VOCAB, DIM_WordEmbedding])
    embedding_init = E.assign(embedding_placeholder)
    rnn_inputs = tf.nn.embedding_lookup(E, train_x) #vectors after embedding

    # RNN LAYER
    cell = tf.nn.rnn_cell.BasicRNNCell(DIM_WordEmbedding)
    seqlen = tf.placeholder(tf.int32, [BATCH_SIZE])
    #TODO
    np_state = None
    init_state_placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, DIM_WordEmbedding])

    #init_state = tf.Variable(tf.constant(0.0, shape = [BATCH_SIZE, DIM_WordEmbedding]), trainable=False, name = 'init_state')
    init_state = tf.get_variable('init_state', shape = [BATCH_SIZE, DIM_WordEmbedding],trainable=False)
    init_assign = tf.assign(init_state, init_state_placeholder)
    #init_state = tf.get_variable('init_state', [BATCH_SIZE, DIM_WordEmbedding])
    rnn_outputs, init_state = tf.nn.dynamic_rnn(cell, tf.nn.dropout(rnn_inputs, 0.8), sequence_length = seqlen, initial_state = init_state, time_major = False) #TODO: feed in the np_state and store back
    # Mask
    Mplaceholder = tf.placeholder(tf.float32, [BATCH_SIZE, DIM_RNN])
    rnn_flat_outputs = tf.reshape(rnn_outputs, [-1, DIM_WordEmbedding])

    # HIDDEN LAYER:
    W = tf.Variable(tf.random_uniform([DIM_WordEmbedding, DIM_VOCAB],-1.0, 1.0), name = 'W')
    b = tf.Variable(tf.constant(0.1, shape=[DIM_VOCAB]), name='b')
    learning_rate = tf.placeholder(tf.float32, shape=[])
    u = tf.nn.dropout(tf.add(tf.matmul(tf.nn.dropout(rnn_flat_outputs, 0.8), W), b), 0.8)
    # COST FUNCTION
    y_flat = tf.reshape(sparse_train_y, [-1])
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y_flat, logits = u)
    # Masking over cross_entropy
    flat_mask = tf.reshape(Mplaceholder, [-1])
    # reshape back the loss
    masked_cross_entropy = tf.reshape(flat_mask * cross_entropy, [BATCH_SIZE, DIM_RNN])
    mean_loss_by_example = tf.reduce_sum(masked_cross_entropy, reduction_indices = 1) #/ tf.reduce_sum(Mplaceholder, reduction_indices = 1)
    mean_loss = tf.reduce_mean(mean_loss_by_example)
    # training
    train_step = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(mean_loss)
    init_op = tf.global_variables_initializer()


############################## Testing Graph
#init_state = tf.get_variable('init_state', shape = [BATCH_SIZE, DIM_WordEmbedding])
with tf.variable_scope('all', reuse = True):
    #init_state = tf.get_variable(name = 'init_state', shape = [BATCH_SIZE, DIM_WordEmbedding])
    rnn_outputs_valid, init_state = tf.nn.dynamic_rnn(cell, rnn_inputs, sequence_length = seqlen, initial_state = init_state, time_major = False)
    rnn_outputs_valid_flat = tf.reshape(rnn_outputs_valid, [-1, DIM_WordEmbedding])
    u_valid = tf.add(tf.matmul(rnn_outputs_valid_flat, W), b)
    p_valid = tf.nn.softmax(u_valid)
    mask_flat_valid = tf.reshape(Mplaceholder, shape = [-1])
    y_flat_valid = tf.reshape(sparse_train_y, shape = [-1])
    

#####################################

# start session and init
sess = tf.Session()
sess.run(init_op)
sess.run(embedding_init, feed_dict = {embedding_placeholder: np.matrix(Mword)})

#recording accuracy     
outfile = open ('results.txt', 'w')


for i in range(EPOCH):
    print ('------------------running ' + str(i)+ 'th epoch --------------------')
    print ('shuffling batch')
    sf = SF(training_data, BATCH_SIZE)
    sf.shuffle()
    ''' RESOLVED ISSUE: 101 SIZE (SHALLOW COPY)
        for tup in sf.get_batch():
        j, batch_xs, batch_ys, batch_lens = tup
        num_sentences = len(batch_xs[0])
        for ns in range(num_sentences):
            #labels_batch = np.array(getLabelsBatch(batch_ys, ns, sparse = True))
            lens_batch = np.array(getSentenceLength(batch_lens, ns))
            s = np.array(getSentenceBatch(batch_xs, ns))
            if (s.shape[1] == 100):
                print (s.shape)               
    sys.exit()'''
    for tup in sf.get_batch(): # getting a batch of *documents*
        j, batch_xs, batch_ys, batch_lens = tup  # note: j means the j'th batch
        print ('training the ', str(j), 'th batch')
        num_sentences = len(batch_xs[0])
        # np_state for initialization at start of doc: [BATCH_SIZE, DIM_WordEmbedding]
        np_state = sess.run(cell.zero_state(BATCH_SIZE, dtype = tf.float32))
        sess.run(init_assign, feed_dict={init_state_placeholder:np_state})
        
        #for ns in range(num_sentences):
        for ns in range(3):
            # prepare the batch of sentences, label, sentence_lens
            sentences_batch = np.array(getSentenceBatch(batch_xs, ns))
            labels_batch = np.array(getLabelsBatch(batch_ys, ns, sparse = True))
            lens_batch = np.array(getSentenceLength(batch_lens, ns))
            mask_out = getMask(lens_batch)
            #print (sentences_batch.shape)
            sess.run(train_step, feed_dict={train_x: sentences_batch, sparse_train_y: labels_batch, seqlen: lens_batch, learning_rate: 0.0001, Mplaceholder:mask_out})
            
            
        # validation accuracy
        # for loop to feed in the whole document, with hidden unit updated
        sf_v = SF(validation_data, BATCH_SIZE)
        sf.shuffle()

        ppl_sum = 0.0
        count_valid = 0

        for tup_validation in sf_v.get_batch():
            print (str(count_valid)+ ' th validation batch')
            count_valid +=1
            j_valid, batch_xs_valid, batch_ys_valid, batch_lens_valid = tup_validation
            # get documents labels: batch_ys_valid DIM:[BS*DS*LS] 
            doc_labels_valid = [doc[0] for doc in batch_ys_valid] 
            np_state_valid = sess.run(cell.zero_state(BATCH_SIZE, dtype = tf.float32))
            sess.run(init_assign, feed_dict={init_state_placeholder:np_state_valid})
            # iterate through the x's and sum to np_x_sums
            for ns in range(num_sentences):
                # prepare the validation set sentences, label, sentence_lens in batches
                # average the accuracy after all batches processed
                sen_batch_valid = np.asarray(getSentenceBatch(batch_xs_valid, ns))
                label_batch_valid = np.asarray(getLabelsBatch(batch_ys_valid, ns,sparse = True))
                lens_batch_valid = np.asarray(getSentenceLength(batch_lens_valid, ns))
                mask_out_valid = np.asarray(getMask(lens_batch_valid))
                PF = sess.run(p_valid, feed_dict={train_x:sen_batch_valid, seqlen:lens_batch_valid}) 
                #F for flat
                print (PF)
                MF = sess.run(mask_flat_valid, feed_dict={Mplaceholder: mask_out_valid})
                print (MF)
                YF = sess.run(y_flat_valid, feed_dict={sparse_train_y:label_batch_valid})
                print (YF)
                #sys.exit()
                for index in range(len(PF)):
                    ppl_sum += PF[index][YF[index]]*MF[index]
        # compute the perplexity:
        print (ppl_sum)
        ppl = math.exp((-1/WORDCOUNTS[1])*ppl_sum)
        print (ppl)
        outfile.write(str(ppl_sum) + ' ' + str(ppl)+'\n')
        outfile.flush()









