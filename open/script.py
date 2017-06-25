# script for running iterations of training

import numpy as np
import os
from random import shuffle
import re
import tensorflow as tf

import urllib.request
import zipfile
import lxml.etree

# For now, we're only interested in the subtitle text, so let's extract that from the XML:
with zipfile.ZipFile('ted_en-20160408.zip', 'r') as z:
    doc = lxml.etree.parse(z.open('ted_en-20160408.xml', 'r'))
input_text = doc.xpath('//content/text()')

    
for i in range(len(input_text)):
    # for each input_text, remove the parenthesise
    input_text[i] = re.sub(r'\([^)]*\)', '', input_text[i])
    # substitute \n characters with spaces
    input_text[i] = input_text[i].split('\n')
    #Now, let's attempt to remove speakers' names that occur at the beginning of a line, by deleting pieces of the form "<up to 20 characters>:", as shown in this example. Of course, this is an imperfect heuristic.
    for j in range(len(input_text[i])):
        m = re.match(r'^(?:(?P<precolon>[^:]{,20}):)?(?P<postcolon>.*)$', input_text[i][j])
        input_text[i][j] = ''.join([sent for sent in m.groupdict()['postcolon'].split('.') if sent])
        
for i in range(len(input_text)):
    for j in range(len(input_text[i])):
        tokens = re.sub(r"[^a-z0-9]+", " ", input_text[i][j].lower()).split()
        input_text[i][j] = tokens

from gensim.models import Word2Vec
# import the script which trains and saves word2Vec model
import trainWord2Vec 

# helper function 1: loading W2V model
def getWord2Vec():
    # execute python script to train a word2Vec model
    sentences = []
    for text in input_text:
        sentences += text
    trainWord2Vec.train(sentences)
    return None
        
# Mword as the initialized word embedding
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




