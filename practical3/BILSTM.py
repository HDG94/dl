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

# initialize the dimension variables
DIM_VOCAB = 0
DIM_WordEmbedding = 100
DIM_Label = 8
DIM_Hidden = 100
DIM_RNN = 100
BATCH_SIZE = 50
EPOCH = 10
PADDING = 'pad0'
UNKNOWN = 'UKN'
DROPOUT = 0.8

# Download the dataset if it's not already there: this may take a minute as it is 75MB
if not os.path.isfile('ted_en-20160408.zip'):
    urllib.request.urlretrieve("https://wit3.fbk.eu/get.php?path=XML_releases/xml/ted_en-20160408.zip&filename=ted_en-20160408.zip", filename="ted_en-20160408.zip")
    
# For now, we're only interested in the subtitle text, so let's extract that from the XML:
with zipfile.ZipFile('ted_en-20160408.zip', 'r') as z:
    doc = lxml.etree.parse(z.open('ted_en-20160408.xml', 'r'))
input_text = doc.xpath('//content/text()')
#print (len(input_text))

input_labels = doc.xpath('//keywords/text()')
#print (len(input_labels))
#print (input_labels[0])

# LABELS = ['ooo', 'Too', 'oEo', 'ooD', 'TEo', 'ToD', 'oED', 'TED']
# Assign labels to each document
labels = ['' for i in range(len(input_labels))]
count = 0
countT = 0
countE = 0
countD = 0
for kw in input_labels:
    l = ['o', 'o', 'o']
    if 'technology' in kw:
        l[0] = 'T'
        countT += 1
    if 'entertainment' in kw:
        l[1] = 'E'
        countE += 1
    if 'design' in kw:
        l[2] = 'D'
        countD += 1
    labels[count] = ''.join(l)
    count += 1
    
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
    
# doc: input_text[i], which is a list of sentences (list of words)

from prepare_sentences import Preparer
P = Preparer(DIM_RNN, input_text)
docLengths = P.cutDocs() # docLengths are the list of unpadded lengths of each sentence
# we add the label and length information to the data
# prepare labels for each sentence for each doc
sentence_labels = []
for i in range(len(labels)):
    # for each document, assign labels to all sentences
    num_sen = len(input_text[i])
    sentence_labels.append([labels[i]]*num_sen)

# IMPORTANT - format of data:
# data[0]: list of documents (list of fixed-size sentences (list of words))
# data[1]: list of (list of labels for each sentence) for each doc
# data[2]: list of (sentence lengths for each sentence) for each doc
text_labels = [[input_text[i], sentence_labels[i], docLengths[i]] for i in range(len(input_text))]
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
if (embeddingName == 'Word2Vec'):
    DIM_VOCAB, DIM_WordEmbedding = w2v_model.syn0.shape

Mword = np.concatenate((Mword, [[0.0]*DIM_WordEmbedding]), axis = 0)
Mword = np.concatenate((Mword, [[0.0]*DIM_WordEmbedding]), axis = 0)
print (Mword.shape)
word_index = w2v_model.index2word
word_index.append(PADDING)
word_index.append(UNKNOWN)
DIM_VOCAB += 1
DIM_VOCAB += 1
    
import tensorflow as tf

# construct a word dictionary, word - index
word_dic = {}
for w in word_index:
    word_dic[w] = len(word_dic)
print (len(word_dic))

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
    
one_hots = [[0]*8 for i in range(8)]
LABELS = ['ooo', 'Too', 'oEo', 'ooD', 'TEo', 'ToD', 'oED', 'TED']
for i in range(8):
    one_hots[i][i] = 1 
label_dic = dict(zip(LABELS, one_hots))

# change the labels to each sentence
train_y_vec = [0]*len(training_data)
validation_y_vec = [0]*len(validation_data)
test_y_vec = [0]*len(testing_data)

for i in range(len(training_data)):
    lab = label_dic[training_data[i][1][0]]
    training_data[i][1] = [lab]*len(training_data[i][1])

for i in range(len(validation_data)):
    lab = label_dic[validation_data[i][1][0]]
    validation_data[i][1] = [lab]*len(validation_data[i][1])
for i in range(len(testing_data)):
    lab = label_dic[testing_data[i][1][0]]
    testing_data[i][1] = [lab]*len(testing_data[i][1])
    

###################
### RNN NETWORK ###
###################

# Helper functions
def getMask(lens):
    # lens: the list of lengths of the sequence (max = DIM_RNN)
    Vones = [1.0]*DIM_WordEmbedding
    Vzeros = [0.0]*DIM_WordEmbedding
    mask = []
    for i in range(BATCH_SIZE):
        temp = [Vones if j<lens[i] else Vzeros for j in range(DIM_RNN)]
        mask.append(temp)
    return np.array(mask)
# For batching
def getSentenceBatch(batch, ns):
    # batch is the list of documents
    # we need to get the ns'th sentence of each batch, return dim: [BS*DIM_RNN]
    return [doc[ns] for doc in batch]    
def getSentenceLabels(label_batch, ns):
    # batch is the list of documents, each containing one label for each sentence
    # we need to get the ns'th sentence's label of each batch, return dim: [BS*DIM_Label]
    return [doc[ns] for doc in label_batch]
def getSentenceLength(lens_batch, ns):
    # batch is the list of documents, each containing one length for each sentence
    # we need a list of lengths for each sentence in the batch, return dim: [BS]
    return [doc[ns] for doc in lens_batch] 

from shuffle import shuffle as SF

# INPUT LAYER
with tf.variable_scope('all', reuse = False):
    train_x = tf.placeholder(tf.int32, [BATCH_SIZE, DIM_RNN], name= 'train_x') #integer list input
    train_y = tf.placeholder(tf.int32, [BATCH_SIZE, DIM_Label], name= 'train_y') #correct labels
    y_ = tf.placeholder(tf.int32, [BATCH_SIZE, DIM_Label], name = 'y_') #predicted labels

    # EMBEDDING LAYER
    E = tf.Variable(tf.constant(0.0, shape = [DIM_VOCAB, DIM_WordEmbedding]), trainable=True, name = 'E')
    embedding_placeholder = tf.placeholder(tf.float32, [DIM_VOCAB, DIM_WordEmbedding])
    embedding_init = E.assign(embedding_placeholder)
    rnn_inputs = tf.nn.embedding_lookup(E, train_x) #vectors after embedding

    # RNN LAYER
    #cell = tf.nn.rnn_cell.GRUCell(DIM_WordEmbedding)
    cell = tf.nn.rnn_cell.BasicLSTMCell(DIM_WordEmbedding, state_is_tuple=True)
    cell_back = tf.nn.rnn_cell.BasicLSTMCell(DIM_WordEmbedding, state_is_tuple=True)
    seqlen = tf.placeholder(tf.int32, [BATCH_SIZE])
	#TODO
    np_state = None
    #init_state_placeholder = tf.placeholder(tf.float32, shape = [2, BATCH_SIZE, DIM_WordEmbedding])
    #init_state = tf.get_variable('init_state', shape = [2, BATCH_SIZE, DIM_WordEmbedding], trainable = False)
    #init_assign = tf.assign(init_state, init_state_placeholder)
    init_state_c_fw = tf.Variable(cell.zero_state(BATCH_SIZE, dtype=tf.float32)[0], trainable = False)
    init_state_h_fw = tf.Variable(cell.zero_state(BATCH_SIZE, dtype=tf.float32)[1], trainable = False)
    init_state_c_bw = tf.Variable(cell_back.zero_state(BATCH_SIZE, dtype=tf.float32)[0], trainable = False)
    init_state_h_bw = tf.Variable(cell_back.zero_state(BATCH_SIZE, dtype=tf.float32)[1], trainable = False)
    init_assign_c_fw = init_state_c_fw.assign(cell.zero_state(BATCH_SIZE, dtype=tf.float32)[0])
    init_assign_h_fw = init_state_h_fw.assign(cell.zero_state(BATCH_SIZE, dtype=tf.float32)[1])
    init_assign_c_bw = init_state_c_bw.assign(cell_back.zero_state(BATCH_SIZE, dtype=tf.float32)[0])
    init_assign_h_bw = init_state_h_bw.assign(cell_back.zero_state(BATCH_SIZE, dtype=tf.float32)[1])
    
    (rnn_outputs_fw, rnn_outputs_bw), ((init_state_c_fw, init_state_h_fw), (init_state_c_bw, init_state_h_bw)) = tf.nn.bidirectional_dynamic_rnn(cell, cell_back, tf.nn.dropout(rnn_inputs, DROPOUT), sequence_length = seqlen, initial_state_fw = tf.nn.rnn_cell.LSTMStateTuple(init_state_c_fw, init_state_h_fw), initial_state_bw = tf.nn.rnn_cell.LSTMStateTuple(init_state_c_bw, init_state_h_bw), time_major = False) #TODO: remember to feed in the np_state and store back
    rnn_outputs = rnn_outputs_fw + rnn_outputs_bw
    # Mask
    M = tf.Variable(tf.constant(0.0, shape=[BATCH_SIZE, DIM_RNN, DIM_WordEmbedding]), trainable = False)
    Mplaceholder = tf.placeholder(tf.float32, [BATCH_SIZE, DIM_RNN, DIM_WordEmbedding])
    Massign = tf.assign(M, Mplaceholder) #Add 'assign' to each iteration before run
    rnn_masked_outputs = tf.mul(M, rnn_outputs)

	# x layer after summation
    x = tf.reduce_sum(rnn_masked_outputs, reduction_indices= 1) # reduce to dim(BS*DIM_WE)
    x_placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, DIM_WordEmbedding])

	# HIDDEN LAYER:
    W = tf.Variable(tf.random_uniform([DIM_WordEmbedding, DIM_Hidden],-1.0, 1.0), name = 'W')
    b = tf.Variable(tf.constant(0.1, shape=[DIM_Hidden]), name='b')
    V = tf.Variable(tf.random_uniform([DIM_Hidden, DIM_Label], -1.0, 1.0), name = 'V')
    c = tf.Variable(tf.constant(0.1, shape=[DIM_Label]), name='c')
    learning_rate = tf.placeholder(tf.float32, shape=[])
    h = tf.tanh(tf.add(tf.matmul(tf.nn.dropout(x,DROPOUT), W), b))
    u = tf.add(tf.matmul(h, V), c)

    p = tf.nn.softmax(logits = u)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = u, labels = train_y))
    train_step = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cross_entropy)
    init_op = tf.global_variables_initializer()

with tf.variable_scope('all', reuse = True):
    ####################### Testing Graph
    # RNN LAYER:
    (rnn_outputs_valid_fw, rnn_outputs_valid_bw), ((init_state_c_fw, init_state_h_fw),(init_state_c_bw, init_state_h_bw)) = tf.nn.bidirectional_dynamic_rnn(cell, cell_back, rnn_inputs, sequence_length = seqlen, initial_state_fw = tf.nn.rnn_cell.LSTMStateTuple(init_state_c_fw, init_state_h_fw), initial_state_bw = tf.nn.rnn_cell.LSTMStateTuple(init_state_c_bw, init_state_h_bw), time_major = False)
    rnn_outputs_valid = rnn_outputs_valid_fw + rnn_outputs_valid_bw
    rnn_masked_outputs_valid = tf.mul(M, rnn_outputs_valid)

	# x layer after summation
    x_valid = tf.reduce_sum(rnn_masked_outputs_valid, reduction_indices= 1) # reduce to dim(BS*DIM_WE)
	# HIDDEN LAYER:
    h_valid = tf.tanh(tf.add(tf.matmul(x_placeholder, W), b))
    u_valid = tf.add(tf.matmul(h_valid, V), c)
    p_valid = tf.nn.softmax(logits = u_valid)
	#####################################

# start session and init
sess = tf.Session()
sess.run(init_op)
sess.run(embedding_init, feed_dict = {embedding_placeholder: np.matrix(Mword)})

ac_train_list = []
ac_validation_list = []
   
outfile = open ('resultsBIDLSTM.txt', 'w')
for i in range(EPOCH):
    print ('------------------running ' + str(i)+ 'th epoch --------------------')
    print ('shuffling batch')
    sf = SF(training_data, BATCH_SIZE)
    sf.shuffle()
    
    for tup in sf.get_batch(): # getting a batch of *documents*
        j, batch_xs, batch_ys, batch_lens = tup  # note: j means the j'th batch
        #print ('training the ', str(j), 'th batch')
        num_sentences = len(batch_xs[0])
        # np_state for initialization at start of doc: [BATCH_SIZE, DIM_WordEmbedding]
        np_state = sess.run(cell.zero_state(BATCH_SIZE, dtype=tf.float32))
        #sess.run(init_assign, feed_dict={init_state_placeholder:np_state})
        sess.run(init_assign_c_fw)
        sess.run(init_assign_h_fw)
        sess.run(init_assign_c_bw)
        sess.run(init_assign_h_bw)
        for ns in range(num_sentences):
            # prepare the batch of sentences, label, sentence_lens
            sentences_batch = getSentenceBatch(batch_xs, ns)
            labels_batch = getSentenceLabels(batch_ys, ns)
            lens_batch = getSentenceLength(batch_lens, ns)
            st = np.array(sentences_batch)
            mask_out = getMask(lens_batch)
            sess.run(Massign, feed_dict={Mplaceholder:mask_out})
            sess.run(train_step, feed_dict={train_x: np.array(sentences_batch), train_y: np.array(labels_batch), seqlen: np.array(lens_batch), learning_rate: 0.0001})
        
        # validation accuracy
        # for loop to feed in the whole document, with hidden unit updated
        sf_v = SF(validation_data, BATCH_SIZE)
        ac_valid = 0.0 
        count_round = 0 # for average the acc_valid_avg
        for tup_validation in sf_v.get_batch():
            count_round += 1
            j_valid, batch_xs_valid, batch_ys_valid, batch_lens_valid = tup_validation
            # get documents labels: batch_ys_valid DIM:[BS*DS*LS] 
            doc_labels_valid = [doc[0] for doc in batch_ys_valid] 
            np_state_valid = sess.run(cell.zero_state(BATCH_SIZE, dtype=tf.float32))
            #sess.run(init_assign, feed_dict={init_state_placeholder:np_state_valid})
            sess.run(init_assign_c_fw)
            sess.run(init_assign_h_fw)
            sess.run(init_assign_c_bw)
            sess.run(init_assign_h_bw)
            np_x_sums = np.zeros(shape= [BATCH_SIZE, DIM_WordEmbedding], dtype = np.float32)
            # iterate through the x's and sum to np_x_sums
            for ns in range(num_sentences):
                # prepare the validation set sentences, label, sentence_lens in batches
                # average the accuracy after all batches processed
                sen_batch_valid = getSentenceBatch(batch_xs_valid, ns)
                label_batch_valid = getSentenceLabels(batch_ys_valid, ns)
                lens_batch_valid = getSentenceLength(batch_lens_valid, ns)
                mask_out_valid = getMask(lens_batch_valid)
                sess.run(Massign, feed_dict={Mplaceholder:mask_out_valid})
                x_temp = sess.run(x_valid, feed_dict={train_x: sen_batch_valid, train_y: label_batch_valid, seqlen: lens_batch_valid, learning_rate: 0.0001})
                np_x_sums += x_temp
            
            correct_prediction = tf.equal(tf.argmax(p_valid, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            ac_validation = sess.run(accuracy, feed_dict={ y_: np.matrix(doc_labels_valid),x_placeholder: np_x_sums})
            ac_valid += ac_validation
        if count_round>0:
            ac_valid /= count_round
            print (ac_valid)
        else:
            ac_valid = ac_valid
            print (ac_valid)
        outfile.write(str(ac_valid)+'\n')
        outfile.flush()









