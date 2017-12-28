from nltk.tokenize import word_tokenize
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import itertools

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

data = load_doc("cerpen.txt")
data = data.lower()
data = data.split('\n')
for i in range(len(data)):
    data[i] = word_tokenize(data[i])

# create mapping of unique chars to integers
token = list(itertools.chain(*data))
token = sorted(list(set(token)))
dictionary = dict((c, i) for i, c in enumerate(token, 1))
dictionary[''] = 0

word2int = dictionary
int2word = dict((c, i) for i, c in dictionary.items())

dataX = []
for i in range(0, len(data)):
    dataX.append([dictionary[char] for char in data[i]])

maxlen = 0
for i in range(len(dataX)):
    if maxlen < len(dataX[i]):
        maxlen = len(dataX[i])

print(maxlen)
# dataY = [None] * len(dataX)
# for i in range(len(dataX)):
#     dataY[i] = []
#     for j in range(len(dataX[i])-1):
#         dataY[i].append(dataX[i][j+1])

dataNPX = pad_sequences(dataX, maxlen=maxlen)
# dataNPX = np.zeros((len(dataX),maxlen,1), dtype=np.int)
# for i in range(0,len(dataX)):
#     for j in range(0,len(dataX[i])):
#         dataNPX[i][j] = dataX[i][j]

# dataNPY = np.zeros((len(dataX),maxlen,1), dtype=np.int)
# for i in range(0,len(dataY)):
#     for j in range(0,len(dataY[i])):
#         dataNPY[i][j] = dataY[i][j]

# dataNPX = np.reshape(dataNPX,(len(data),maxlen))
# dataNPY = np.reshape(dataNPY,(len(data),maxlen))

x = [None] * 5
x[0] = dataNPX
x[1] = dataX                  
x[2] = data
x[3], x[4]    = word2int, int2word

import pickle
with open('cerpen.p', 'wb') as handle:
    pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)