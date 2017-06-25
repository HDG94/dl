from gensim.models import Word2Vec
import numpy as np
from sklearn.cluster import KMeans
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.plotting import figure, show, output_file
#from bokeh.palettes import Inferno11 as palette
import palette
import itertools


'''To Switch between Ted and Wiki, change:
1. Line 24: wiki_model/ted_model
2. Line 57: allwords/allwords_wiki

'''
# Source of Kmeans & Word2Vec:
# https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-3-more-fun-with-word-vectors

# some useful attributes from word2Vec:
# 1. [model.syn0] The list of word vectors (.shape gives (length, dimension) eg.(14427,100))
# 2. [model.index2word] The list of all words

# load the model
model = Word2Vec.load('wiki_model')
M_words = model.syn0

# first define the properties of the clusters
num_clusters = int(30)
#colors = itertools.cycle(palette)
#colorList = []
colorList = palette.get_palette()

# Kmeans object initialization
print('clustering')
cluster = KMeans(n_clusters = num_clusters)
idx = cluster.fit_predict(M_words) # idx is the cluster assignments for each word (size 14427 array)

# Now map the dictionaries with the words to map each word to a cluster number
word_cluster_dic = dict(zip(model.index2word, idx))

# assign a color for each cluster
count = 0
'''for c in colors:
    count +=1
    colorList.append(c)
    if count== num_clusters:
        break'''

# map the color of clusters to each data point
colors = [colorList[i] if i<len(colorList) else colorList[i%(len(colorList))] for i in idx]



################ plotting the clusters ##########################
# load the top words of ted
words_top_ted = []
with open('allwords_wiki', 'r') as Fwords:
    for l in Fwords:
        words_top_ted. append(l.split(' ')[0])
words_top_ted = words_top_ted[:1000]

# This assumes words_top_ted is a list of strings, the top 1000 words
words_top_vec_ted = model[words_top_ted]
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)
words_top_ted_tsne = tsne.fit_transform(words_top_vec_ted)

p = figure(tools="pan,wheel_zoom,reset,save",toolbar_location="above",title="word2vec K-means Clustering for most common words")

source = ColumnDataSource(data=dict(x1=words_top_ted_tsne[:,0],x2=words_top_ted_tsne[:,1], names=words_top_ted))

p.scatter(x="x1", y="x2", size=8, source=source, fill_color = colors)

labels = LabelSet(x="x1", y="x2", text="names", y_offset=6,text_font_size="8pt", text_color="#555555",source=source, text_align='center')
p.add_layout(labels)

show(p)



