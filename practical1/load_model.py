from gensim.models import Word2Vec
import numpy as np
from numpy import linalg as LA

''' To switch to wiki, change the line below:
	load('wiki_model')
'''

# load the ted model saved earlier
model_ted = Word2Vec.load('ted_model')
# find some words with interesting neighbors
model_ted.most_similar(positive = ['language'])

''' Result Terminal Outputs:
[('mathematics', 0.6643838882446289),
 ('culture', 0.659943163394928),
 ('english', 0.6539066433906555),
 ('logic', 0.6274464130401611),
 ('narrative', 0.6246222257614136),
 ('nature', 0.6029856204986572),
 ('domain', 0.5986967086791992),
 ('desire', 0.592326283454895),
 ('meaning', 0.588565468788147),
 ('empathy', 0.5859334468841553)]

'''
# task for retrieving two word vectors and compute the distance
v1 = model_ted['computer']
v2 = model_ted['technology']
distance = np.dot(v1, v2)
distance = distance/LA.norm(v1)/LA.norm(v2)
#compute the distance computed by gensim
print ()
print ('---------------Distance Comparison----------------')
print ('Gensim Distance', str(model_ted.similarity('computer', 'technology')))
print ('Cosine Distance', str(distance))



