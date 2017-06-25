# this is used for shuffling and feeding the data:
import numpy as np
import math

class Shuffle:
    
    def __init__(self, data_label, batch_size, is_training = False):
        self.data_label = data_label
        self.batch_size = batch_size
        self.N = len(data_label) # number of instances
        self.Shuffled = False
        self.is_training = is_training
    
    def shuffle(self):
        # Input: data_label is the tuple (doc, summary, docLens, sumLens)
        print ('data shuffled')
        np.random.shuffle(self.data_label)
        self.Shuffled = True
    
    def get_iteration(self):
        return math.ceil(self.N/self.batch_size)
    
    def get_batch(self):
        # each loop yields one batch of data_label 
        if self.is_training and not (self.Shuffled):
            self.shuffle()
        for i in range(self.get_iteration()):
            start = self.batch_size * i
            end = start + self.batch_size
            if (end < self.N):
                data = [tup[0] for tup in self.data_label[start:end]]
                summary = [tup[1] for tup in self.data_label[start:end]]
                docLens = [tup[2] for tup in self.data_label[start:end]]
                sumLens = [tup[3] for tup in self.data_label[start:end]]
                yield [i, data, summary, docLens, sumLens]
            else:
                # concatenate the two parts
                data = [tup[0] for tup in self.data_label[start:]]
                summary = [tup[1] for tup in self.data_label[start:]]
                docLens = [tup[2] for tup in self.data_label[start:]]
                sumLens = [tup[3] for tup in self.data_label[start:]]
                N2 = end - self.N
                data += [tup[0] for tup in self.data_label[0: N2]]
                summary += [tup[1] for tup in self.data_label[0:N2]]
                docLens += [tup[2] for tup in self.data_label[0: N2]]
                sumLens += [tup[3] for tup in self.data_label[0: N2]]
                yield [i, data, summary, docLens, sumLens]

