from plot import Plot

p = Plot(title = 'Learning Curve Dropout')
fileNames =  ['Word2VecH150.txt', 'DROPWord2VecH150.txt', '2HiddenWord2VecH100.txt', 'DROP_2HiddenWord2VecH100.txt']
seriesNames =  ['H150', 'H150 Dropout', '2H*100', '2H*100 Dropout']
p.storeList(fileNames, seriesNames)
p.plotSeries()
