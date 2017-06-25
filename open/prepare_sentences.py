# Prepare the sentences into the required length
import math
from config import CONFIG

class Preparer:
    def __init__(self, data, is_summary = False):
        self.data = data
        self.PADDING = 'pad0'
        self.total_wordCount = [0,0,0] # train,valid,test
        self.startstopadded = False
        self.is_summary = is_summary
            
    def addStartStopWords(self, STARTWORD, STOPWORD):
        # now the documents are list of words
        # pad start word and stop words
        for i in range(len(self.data)):
            self.data[i] = [STARTWORD] + self.data[i] + [STOPWORD]
        self.startstopadded = True
    
    def getMaxLength(self):
        # find the max length of all documents
        maxLen = 0
        for doc in self.data:
            leni = len(doc)
            if leni>maxLen:
                maxLen = leni
        return maxLen
        
    def cutDocs(self):
        # cut each doc longer to the length of CONFIG.DIM_RNN - 2
        cliplen = CONFIG.DIM_RNN - 2
        if (self.is_summary):
            cliplen = CONFIG.DIM_DECODER - 2
        for i in range(len(self.data)):
            self.data[i] = self.data[i][:cliplen]
        
    def countDocs(self):
        # first pad the docs by the maxlen
        # and return the lengths of each document
        if not self.startstopadded:
            self.addStartStopWords(CONFIG.STARTWORD, CONFIG.STOPWORD)
        docLengths = [] # docLengths are lists of unpadded lengths of sequences
        maxLen = CONFIG.DIM_RNN
        
        if (self.is_summary):
            maxLen = CONFIG.DIM_DECODER
        for docIndex in range(len(self.data)):
            dl = len(self.data[docIndex])
            if (docIndex < 1585):
                self.total_wordCount[0] += dl-1
            elif (docIndex < 1835):
                self.total_wordCount[1] += dl-1
            else:
                self.total_wordCount[2] += dl-1

            if (len(self.data[docIndex])<maxLen):
                padLength = maxLen - len(self.data[docIndex])
                self.data[docIndex] += [self.PADDING]*padLength

            if (self.is_summary):
                docLengths.append(dl-1)
            else:
                docLengths.append(dl)

        return docLengths
    
    def doc2Int(self):
        for i in range(len(self.data)):
            tokens_int = []
            for j in range(len(self.data[i])):
                if (self.data[i][j] in CONFIG.WORDS):
                    tokens_int.append(CONFIG.WORDS[self.data[i][j]])
                else:
                    tokens_int.append(CONFIG.WORDS[CONFIG.UNKNOWN])
            self.data[i] = tokens_int
        
    def getTotalWordCount(self):
        return self.total_wordCount
