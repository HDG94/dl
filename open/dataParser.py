import sys
import re
import os
import urllib.request
import zipfile
import lxml.etree
import json

class DataParser(object):
    def __init__(self):
        self.input_text = None
        self.summaries = None
        
    def parseFile(self):
        # Download the dataset if it's not already there: this may take a minute as it is 75MB
        if not os.path.isfile('ted_en-20160408.zip'):
            urllib.request.urlretrieve("https://wit3.fbk.eu/get.php?path=XML_releases/xml/ted_en-20160408.zip&filename=ted_en-20160408.zip", filename="ted_en-20160408.zip")
            
        # For now, we're only interested in the subtitle text, so let's extract that from the XML:
        with zipfile.ZipFile('ted_en-20160408.zip', 'r') as z:
            doc = lxml.etree.parse(z.open('ted_en-20160408.xml', 'r'))
        input_text = doc.xpath('//content/text()')
        summaries = doc.xpath('//description/text()')
            
        for i in range(len(input_text)):
            # for each input_text, remove the parenthesise
            input_text[i] = re.sub(r'\([^)]*\)', '', input_text[i])
            # the input_text contains a lot of new lines
            temp = []
            input_text[i] = ''.join(input_text[i].split('\n'))
                
        temp = []
        count = 0
        for block in input_text:
            m = re.match(r'^(?:(?P<precolon>[^:]{,20}):)?(?P<postcolon>.*)$', block)
            temp.append([sent for sent in m.groupdict()['postcolon'].split('.') if sent])
        input_text = temp

        for i in range(len(input_text)):
            for j in range(len(input_text[i])):
                tokens = re.sub(r"[^a-z0-9]+", " ", input_text[i][j].lower()).split()
                input_text[i][j] = tokens
            temp = []
            for j in range(len(input_text[i])):
                temp.extend(input_text[i][j])
            input_text[i] = temp   
        for i in range(len(summaries)):
            # delete the prefix 'TED Talk Subtitles and Transcript: '
            if len(summaries[i])>0 and summaries[i][0] == 'T':
                if ':' in summaries[i]:
                    summaries[i] = summaries[i][summaries[i].index(':')+2:]
        for i in range(len(summaries)):
            tokens = re.sub(r"[^a-z0-9]+", " ", summaries[i].lower()).split()
            summaries[i] = tokens
            
        self.summaries = summaries
        self.input_text = input_text
        return [input_text, summaries]
        
