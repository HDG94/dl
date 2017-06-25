class CONFIG(object):
# The Constants
    BATCH_SIZE = 200
    EPOCH = 100
    DIM_VOCAB = 0
    DIM_WordEmbedding = 300
    DIM_RNN = 50 #encoder side rnn
    DIM_DECODER = 10
    PADDING = 'pad0'
    UNKNOWN = 'UKN'
    STARTWORD = 'STARTWORD'
    STOPWORD = 'STOPWORD'
    WORDFILE = 'allwords'
    WORDMODELFILE = 'ted_model'
    LEARNINGRATE = 0.001
    KEEPPROB = 0.2
    WORDCOUNTS = []
    WORDS = {}
    WORDSLIST = []
    w2vMatrix = None
