# this is used for shuffling and feeding the data:
import numpy as np
import math

class shuffle:
    
    def __init__(self, data_label, batch_size):
        self.data_label = data_label
        self.batch_size = batch_size
        self.N = len(data_label) # number of instances
        self.DIM_Label = len(data_label[0][1]) # in practical2: dim(label) = 8
        self.Shuffled = False
    
    def shuffle(self):
        # Input: data_label is the tuple (data, label)
        # Input Dimension: data_label = [(data_vec0, label_vec0), (data_vec1, label_vec1), ... num_of_data]
        print ('data shuffled')
        np.random.shuffle(self.data_label)
        self.Shuffled = True
    
    def get_iteration(self):
        return math.ceil(self.N/self.batch_size)
    
    def get_batch(self):
        # each loop yields one batch of data_label 
        if not (self.Shuffled):
            self.shuffle()
        for i in range(self.get_iteration()):
            start = self.batch_size * i
            end = start + self.batch_size
            if (end < self.N):
                data = [tup[0] for tup in self.data_label[start:end]]
                label = [tup[0] for tup in self.data_label[start:end]]
                lens = [tup[1] for tup in self.data_label[start:end]]
                yield [i, data, label, lens]
            else:
                # concatenate the two parts
                data = [tup[0] for tup in self.data_label[start:]]
                label = [tup[0] for tup in self.data_label[start:]]
                lens = [tup[1] for tup in self.data_label[start:]]
                N2 = end - self.N
                data += [tup[0] for tup in self.data_label[0: N2]]
                label += [tup[0] for tup in self.data_label[0:N2]]
                lens += [tup[1] for tup in self.data_label[0: N2]]
                yield [i, data, label, lens]

