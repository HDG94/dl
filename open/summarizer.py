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
from shuffle import Shuffle as SF
from prepare_sentences import Preparer
from dataParser import DataParser
from config import CONFIG
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.rnn import LSTMCell, GRUCell, BasicRNNCell
    
class SummaryModel(object):
    
    def output_fn(self, outputs):
        #return tf.contrib.layers.linear(outputs, CONFIG.DIM_VOCAB)
        return tf.contrib.layers.fully_connected(inputs=outputs, num_outputs=CONFIG.DIM_VOCAB)
    def project_encoder_last_states(self, encoder_last_states):
        return tf.contrib.layers.linear(encoder_last_states, 2* CONFIG.DIM_WordEmbedding)
    def __init__(self, is_training):
        
        ''' Init all constants, embedding, and cell'''
        self.is_training = is_training
        # load the word embedding, init by random
        #self.embedding = tf.get_variable('embedding',initializer=CONFIG.w2vMatrix, trainable=False)
        self.embedding = tf.get_variable('embedding', shape=[CONFIG.DIM_VOCAB, CONFIG.DIM_WordEmbedding], initializer=tf.random_uniform_initializer(-1.0, 1.0))  
        #self.embedding = tf.get_variable('embedding', shape = [CONFIG.DIM_VOCAB, CONFIG.DIM_WordEmbedding], initializer = tf.contrib.layers.xavier_initializer())
        self.addPlaceholders()
        self.addEncoder()
        self.addDecoder()
        print ('model built')        
        
        if (self.is_training):
            self.decoder_outputs = tf.nn.dropout(self.decoder_outputs, CONFIG.KEEPPROB)
            # this is before projection, try masking here
            
            self.u = self.output_fn(self.decoder_outputs)
            self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_placeholder, logits=self.u)
            #self.cross_entropy = tf.reduce_sum(self.u)
            
            # mask entropy with mask
            self.masked_cross_entropy = self.cross_entropy * self.Mplaceholder
            mean_loss_by_example = tf.reduce_sum(self.masked_cross_entropy, reduction_indices = -1)
            self.mean_loss = tf.reduce_mean(mean_loss_by_example)
            
            #self.train_op = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.mean_loss)
            optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
            #self.var_grad = tf.gradients(self.mean_loss, [self._decoder_in_state])[0]
            #gvs = tf.gradients(self.mean_loss)
            gvs = optimizer.compute_gradients(self.mean_loss)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            self.train_op = optimizer.apply_gradients(capped_gvs)
        else:
            self.output_words = tf.cast(tf.argmax(self.decoder_outputs, axis = -1),tf.int32)
             
  
    def last_relevant(self, outputs, length):
        batch_size = CONFIG.BATCH_SIZE
        max_length = CONFIG.DIM_RNN
        out_size = CONFIG.DIM_WordEmbedding
        index = tf.range(0, batch_size) * max_length + tf.maximum(tf.zeros(shape=length.get_shape(), dtype=tf.int32), (length-1))
        flat = tf.reshape(outputs, [-1, out_size])
        relevant = tf.gather(flat, index)
        return relevant
    
    # helper functions    
    def addEncoder(self):
        print ('adding encoder...')
        self.encoder_inputs = tf.nn.embedding_lookup(self.embedding, self.x_placeholder)
        if (self.is_training):
            self.encoder_inputs = tf.nn.dropout(self.encoder_inputs, CONFIG.KEEPPROB)
        cell_fw = BasicRNNCell(CONFIG.DIM_WordEmbedding)
        cell_bw = BasicRNNCell(CONFIG.DIM_WordEmbedding)
        (encoder_output_fw, encoder_output_bw), (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.encoder_inputs, dtype=tf.float32, sequence_length=self.x_lens)
        # concatenate the bidirectional inputs
        self._encoder_outputs = tf.concat([encoder_output_fw, encoder_output_bw],2)
        _decoder_in_state = self.last_relevant(encoder_output_bw, self.x_lens)
        self._decoder_in_state = self.project_encoder_last_states(_decoder_in_state)
        #self._decoder_in_state = tf.concat([fw_state, bw_state],1)
        self.attention_states = self._encoder_outputs
        #self._decoder_in_state = tf.zeros(shape=[CONFIG.BATCH_SIZE, CONFIG.DIM_WordEmbedding *2], dtype=tf.float32)
        #self.attention_states = tf.zeros(shape=[CONFIG.BATCH_SIZE, CONFIG.DIM_RNN, CONFIG.DIM_WordEmbedding * 2])
    def addDecoder(self):
        print ('adding decoder...')
        cell = BasicRNNCell(2* CONFIG.DIM_WordEmbedding)
        self.attention_states = self._encoder_outputs
        self.decoder_inputs_embedded = tf.nn.embedding_lookup(self.embedding, self.y_placeholder)
        # prepare attention: 
        (attention_keys, attention_values, attention_score_fn, attention_construct_fn) = seq2seq.prepare_attention(attention_states=self.attention_states, attention_option='bahdanau', num_units=2* CONFIG.DIM_WordEmbedding)
        
        if (self.is_training):           
            # new Seq2seq train version
            self.check_op = tf.add_check_numerics_ops()
            decoder_fn_train = seq2seq.attention_decoder_fn_train(encoder_state=self._decoder_in_state, attention_keys=attention_keys, attention_values=attention_values, attention_score_fn=attention_score_fn, attention_construct_fn=attention_construct_fn, name='attention_decoder')
            (self.decoder_outputs_train, self.decoder_state_train, self.decoder_context_state_train) = seq2seq.dynamic_rnn_decoder(cell=cell, decoder_fn=decoder_fn_train, inputs=self.decoder_inputs_embedded, sequence_length=self.y_lens, time_major=False)
            self.decoder_outputs = self.decoder_outputs_train
     
        else:
            
            # new Seq2seq version
            start_id = CONFIG.WORDS[CONFIG.STARTWORD]
            stop_id = CONFIG.WORDS[CONFIG.STOPWORD]
            decoder_fn_inference = seq2seq.attention_decoder_fn_inference(encoder_state=self._decoder_in_state, attention_keys = attention_keys, attention_values=attention_values, attention_score_fn=attention_score_fn, attention_construct_fn=attention_construct_fn, embeddings=self.embedding, start_of_sequence_id=start_id, end_of_sequence_id = stop_id, maximum_length = CONFIG.DIM_DECODER, num_decoder_symbols=CONFIG.DIM_VOCAB, output_fn = self.output_fn )
            (self.decoder_outputs_inference, self.decoder_state_inference, self.decoder_context_state_inference) = seq2seq.dynamic_rnn_decoder(cell=cell, decoder_fn=decoder_fn_inference, time_major=False)
            self.decoder_outputs = self.decoder_outputs_inference
            
    
    def addPlaceholders(self):
        self.x_placeholder = tf.placeholder(tf.int32, [CONFIG.BATCH_SIZE, CONFIG.DIM_RNN])
        self.y_placeholder = tf.placeholder(tf.int32, [CONFIG.BATCH_SIZE, CONFIG.DIM_DECODER-1])
        self.label_placeholder = tf.placeholder(tf.int32, [CONFIG.BATCH_SIZE, None])
        self.x_lens = tf.placeholder(tf.int32, [CONFIG.BATCH_SIZE])
        self.y_lens = tf.placeholder(tf.int32, [CONFIG.BATCH_SIZE])
        self.learning_rate = tf.placeholder(tf.float32)
        self.Mplaceholder = tf.placeholder(tf.float32, [CONFIG.BATCH_SIZE, None]) # Mask for the output sequence
    
    def getMask(self, lens, DIM_RNN):
        mask = []
        for i in range(CONFIG.BATCH_SIZE):
            temp = [1.0 if j<lens[i] else 0.0 for j in range(DIM_RNN)]
            mask.append(temp)
        return np.array(mask)        
    
    # run a train step
    def run_train_step(self, sess, doc_batch, summary_batch, label_batch, docLens, summaryLens):
        # note: cut the mask and label_batch to the max summary length of the batch
        maxlen = max(summaryLens)
        # split the summary batch to [BS, maxlen]
        label_batch = label_batch[:, 0:maxlen]
        decoder_mask = self.getMask(summaryLens, CONFIG.DIM_DECODER-1)
        decoder_mask = decoder_mask[:, 0:maxlen]
        '''print (doc_batch)
        print ('##################')
        print ('\n\n')
        print (summary_batch)
        print ('##################')
        print ('\n\n')
        print (label_batch)
        print ('##################')'''
        #print ('\n\n')
        #print (docLens)
        '''
        print ('##################')
        print ('\n\n')
        print (summaryLens)
        print ('##################')
        print ('\n\n')'''
        feed_dict = {self.x_placeholder: doc_batch, self.y_placeholder:summary_batch, self.label_placeholder:label_batch, self.x_lens:docLens, self.y_lens:summaryLens, self.learning_rate:CONFIG.LEARNINGRATE, self.Mplaceholder:decoder_mask}

        _, mce,  labels, ma = sess.run([self.train_op, self.masked_cross_entropy, self.label_placeholder, self.Mplaceholder ], feed_dict = feed_dict)
        l = sess.run(self.mean_loss, feed_dict = feed_dict)
        
        print ('---------MASKED CROSS ENTROPY------------')
        print (mce)
        print ('-------- NEW LOSS-----------------------')
        print (l)
        '''print ('---------LABELS ------------------------')
        for labelList in labels:
            print (list(CONFIG.WORDSLIST[it] for it in labelList))
        print ('---------MASK---------------------------')
        print (ma)
        print (CONFIG.DIM_VOCAB)'''
        #print (var_grad)

        # return loss
        return 0
                
    def _decodeWord(self, decoderOutInt):
        summaries = []
        for x in range(decoderOutInt.shape[0]):
            words = []
            for y in range(decoderOutInt.shape[1]):
                words.append(CONFIG.WORDSLIST[decoderOutInt[x][y]])
            summaries.append(words)
        for s in summaries:
            print (' '.join(s))
            print ('\n\n')
    
    def run_valid_step(self, sess, doc_batch, summary_batch, label_batch, docLens, summaryLens):
        print ('validating...')
        maxlen = max(summaryLens)
        label_batch = label_batch[:, 0:maxlen]
        decoder_mask = self.getMask(summaryLens, CONFIG.DIM_DECODER-1)
        decoder_mask = decoder_mask[:, 0:maxlen]
        feed_dict = {self.x_placeholder: doc_batch, self.y_placeholder:summary_batch, self.label_placeholder:label_batch, self.x_lens:docLens, self.y_lens:summaryLens, self.learning_rate:CONFIG.LEARNINGRATE, self.Mplaceholder:decoder_mask}
        decoderOutInt = sess.run(self.output_words, feed_dict=feed_dict)
        # decode the int to summary
        self._decodeWord(decoderOutInt)
        
        
def _get_doc_batch(docBatch):
    return np.array(docBatch)
def _get_summary_batch(summaryBatch):
    return np.array([summary[0:len(summaryBatch[0])-1] for summary in summaryBatch])
def _get_label_batch(summaryBatch):
    return np.array([summary[1:] for summary in summaryBatch])
    
def run_epoch(sess, m_train, m_valid, training_data, validation_data):
    iters = 0
    # TODO: shuffle and fetch data
    sf_train = SF(training_data, CONFIG.BATCH_SIZE, is_training = True)
    sf_valid = SF(validation_data, CONFIG.BATCH_SIZE, is_training = False)
    
    for tup in sf_train.get_batch(): 
        print ('Training Batch: ', iters)
        _, doc, summary, docLens, sumLens = tup
        doc_batch = _get_doc_batch(doc)
        summary_batch = _get_summary_batch(summary)
        label_batch = _get_label_batch(summary)       
        loss = m_train.run_train_step(sess, doc_batch, summary_batch, label_batch, np.array(docLens), np.array(sumLens))
        print ('training loss: ', loss)
        it = 0
        for tup_valid in sf_valid.get_batch():
            if (it>0):
                break
            it += 1
            _,doc_valid, summary_valid, docLens_valid, sumLens_valid = tup
            doc_batch_v = _get_doc_batch(doc_valid)
            summary_batch_v = _get_summary_batch(summary_valid)
            label_batch_v = _get_label_batch(summary_valid)
            m_valid.run_valid_step(sess, doc_batch_v, summary_batch_v, label_batch_v, np.array(docLens_valid), np.array(sumLens_valid)) 
        iters += 1
    
        
def _initWordDic(from_word2Vec=False):
# Load the words into WORDS dictionary of CONFIG
    if not (from_word2Vec):
        with open(CONFIG.WORDFILE, 'r') as wordsFile:
            for l in wordsFile:
                word, freq = l.split()
                freq = int(freq)
                if (freq > 10):
                    # only record the words having frequency > 1
                    CONFIG.WORDS[word] = len(CONFIG.WORDS)
                    CONFIG.WORDSLIST.append(word)
                else:
                    #CONFIG.WORDS[word] = len(CONFIG.WORDS)
                    #CONFIG.WORDSLIST.append(word)
                    break        
        # now update the dimension
        CONFIG.WORDS[CONFIG.PADDING] = len(CONFIG.WORDS)
        CONFIG.WORDS[CONFIG.UNKNOWN] = len(CONFIG.WORDS)
        CONFIG.WORDS[CONFIG.STARTWORD] = len(CONFIG.WORDS)
        CONFIG.WORDS[CONFIG.STOPWORD] = len(CONFIG.WORDS)
        CONFIG.DIM_VOCAB = len(CONFIG.WORDS)
        CONFIG.WORDSLIST.extend([CONFIG.PADDING, CONFIG.UNKNOWN, CONFIG.STARTWORD, CONFIG.STOPWORD])    
    else:
        # load the word2vec model
        from gensim.models import Word2Vec
        model_ted = Word2Vec.load(CONFIG.WORDMODELFILE)
        w2vMatrix = model_ted.wv.syn0
        #TODO: add random vectors rep padding, unknown, startword, stopword
        w2vIndex = model_ted.wv.index2word
        for wd in w2vIndex:
            CONFIG.WORDS[wd] = len(CONFIG.WORDS)
            CONFIG.WORDSLIST.append(wd)
        CONFIG.WORDS[CONFIG.PADDING] = len(CONFIG.WORDS)
        CONFIG.WORDS[CONFIG.UNKNOWN] = len(CONFIG.WORDS)
        CONFIG.WORDS[CONFIG.STARTWORD] = len(CONFIG.WORDS)
        CONFIG.WORDS[CONFIG.STOPWORD] = len(CONFIG.WORDS)
        CONFIG.DIM_VOCAB = len(CONFIG.WORDS)
        CONFIG.WORDSLIST.extend([CONFIG.PADDING, CONFIG.UNKNOWN, CONFIG.STARTWORD, CONFIG.STOPWORD])
        append_vec = np.random.uniform(low=0.0, high = 1.0, size=[4,CONFIG.DIM_WordEmbedding])
        # concatenate this vector to the w2vMatrix
        CONFIG.w2vMatrix = np.concatenate((w2vMatrix, append_vec), axis=0)
        
            

def main():
    _initWordDic()
    # parse the data using dataParser
    parser = DataParser()
    docs, summary = parser.parseFile()
    p_doc = Preparer(docs)
    p_summary = Preparer(summary, is_summary=True)
    p_doc.cutDocs()
    p_summary.cutDocs()
    docLens = p_doc.countDocs()
    sumLens = p_summary.countDocs()
    print (max(sumLens))
    #sys.exit()
    p_doc.doc2Int()
    p_summary.doc2Int()
    # docs, docLens, summary, sumLens are the data
    data = list(zip(docs, summary, docLens, sumLens))
    training_data = data[:1585]
    validation_data = data[:1835]
    testing_data = data[1835:] 
    
    ''' FIXING THE DIMENSION ISSUES OF BATCHES
    sf_train = SF(training_data, CONFIG.BATCH_SIZE, is_training = True)
    sf_valid = SF(validation_data, CONFIG.BATCH_SIZE, is_training = False)
    for tup in sf_train.get_batch(): 
        _, doc, summary, docLens, sumLens = tup
        doc_batch = _get_doc_batch(doc)
        summary_batch = _get_summary_batch(summary)
        label_batch = _get_label_batch(summary)
        docLens = np.array(docLens)
        summaryLens = np.array(sumLens)  
        print (doc_batch[0])
        print (summary_batch[0])
        print (label_batch[0])
        print (list(doc for doc in docLens))
        print (list(doc for doc in summaryLens))
        sys.exit()'''
        
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-1, 1)
        with tf.name_scope('Train'):
            with tf.variable_scope('Model', reuse=None, initializer=initializer):
                m = SummaryModel(is_training=True)               
        with tf.name_scope('Valid'):
            with tf.variable_scope('Model', reuse=True, initializer=initializer):
                m_valid = SummaryModel(is_training=False)        
        with tf.name_scope('Test'):
            with tf.variable_scope('Model', reuse=True, initializer=initializer):
                m_test = SummaryModel(is_training=False)
        
        init_op = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.visible_device_list='7'
        sess = tf.Session(config = config)
        sess.run(init_op)
        for epoch in range(CONFIG.EPOCH):
            print ('---------------running ' + str(epoch) + 'th epoch ----------------')
            run_epoch(sess, m, m_valid, training_data, validation_data)

main()
                
            
            
            
        
