# Prepare the sentences into the required length
import math
class Preparer:
    # input only the documents of sentences
    # output the documents of fixed length sequences, 'pad0' if it's padded
    def __init__(self, size, data):
        self.size = size
        self.data = data
        self.merged = False
        self.PADDING = 'pad0'
        self.total_wordCount = [0,0,0] # train,valid,test
    
    def mergeDocs(self):
        # merge all sentences of each doc
        for i in range(len(self.data)):
            # doc now contains a list of sentences, which are separate tokens
            temp = []
            doc = self.data[i]
            for sent in doc:
                temp.extend(sent)
            self.data[i] = temp
        self.merged = True
            
    def addStartStopWords(self, STARTWORD, STOPWORD):
        if not self.merged:
            self.mergeDocs()
        # now the documents are list of words
        # pad start word and stop words
        for i in range(len(self.data)):
            self.data[i] = [STARTWORD] + self.data[i] + [STOPWORD]
    
    def getMaxLength(self):
        if not self.merged:
            self.mergeDocs()
        # find the max length of all documents
        maxLen = 0
        for doc in self.data:
            leni = len(doc)
            if leni>maxLen:
                maxLen = leni
        return maxLen
        
    def cutDocs(self):
        # first pad the docs by the maxlen
        # and return the lengths of each document
        docLengths = [] # docLengths are lists of unpadded lengths of sequences
        maxLen = self.getMaxLength()
        ratio = math.ceil(maxLen/self.size)
        maxLen = ratio * self.size
        
        for docIndex in range(len(self.data)):
            seqLen = []
            dl = len(self.data[docIndex])
            if (docIndex < 1585):
                self.total_wordCount[0] += dl-1
            elif (docIndex < 1835):
                self.total_wordCount[1] += dl-1
            else:
                self.total_wordCount[2] += dl-1
            ratio_unpad = math.floor(dl/self.size) # this is the cut-off point, <= this ratio, all are unpadded sequence, then one line contains padded sequence, afterwards all contains paddings
            if (len(self.data[docIndex])<maxLen):
                padLength = maxLen - len(self.data[docIndex])
                self.data[docIndex] += [self.PADDING]*padLength
            # now cut the documents into self.size
            temp = []
            for i in range(ratio):
                if i<ratio_unpad:
                    # add full length to the seqLen
                    seqLen.append(self.size)
                elif i == ratio_unpad:
                    # add the dl - ratio_unpad * self.size to the length
                    seqLen.append((dl - i*self.size-1))
                else:
                    # add 0 to the length
                    seqLen.append(0)
                start = i * self.size
                end = start + self.size
                temp.append(self.data[docIndex][start:end])
            docLengths.append(seqLen)
            self.data[docIndex] = temp
        return docLengths

    def getTotalWordCount(self):
        return self.total_wordCount
