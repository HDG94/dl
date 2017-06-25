from gensim.models import Word2Vec

def train(sentences):
    model_ted = Word2Vec(sentences,size = 300, min_count = 10)
    model_ted.save('ted_model')
