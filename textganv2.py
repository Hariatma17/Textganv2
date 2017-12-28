from keras.layers import Input, Dense, Reshape, Embedding, Dropout, Flatten, Masking
from keras.layers import TimeDistributed, Activation, RepeatVector
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.models import Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model

import pickle
import numpy as np

# Save Text
def save_text(gen_texts, epoch):
    generated = gen_texts*3342
    generated = np.round(generated)
    generated.astype(int)

    text_file = open("text/Hasil %d.txt" %epoch, "w")
    texts = [None] * n
    for i in range(n):
        texts[i] = list()
        index = generated[i].ravel()
        for j in range(53):
            texts[i].append(int2word[index[j]])
    strings = [None] * n
    for i in range(n):
        strings[i] = ' '.join(texts[i])

    sentence = '\n'.join(strings)
    text_file.write(sentence)
    text_file.close()

#load data
x = pickle.load(open("cerpen.p","rb"))
xtrain = x[0]
word2int, int2word = x[3], x[4]
vocab_size = len(word2int)

vocab_size = len(word2int)
seq_length = xtrain.shape[1]

Xtrain = xtrain/3342
Xtrain = np.expand_dims(Xtrain, axis=2)
print(Xtrain.shape)
# define model generator
#GENERATOR with seq2seq
generator = Sequential()

inputs = Input(shape=(53, 1))
encoded = LSTM(256)(inputs)

decoded = RepeatVector(53)(encoded)
decoded = LSTM(1, return_sequences=True)(decoded)

generator = Model(inputs, decoded)

generator.summary()


#define model discriminator
discriminator = Sequential()
discriminator.add(Conv1D(1, kernel_size=2, input_shape=(53,1)))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dropout(0.5))
discriminator.add(MaxPooling1D(pool_size=2, strides=2))
discriminator.add(Flatten())
discriminator.add(Dense(1, activation='sigmoid'))

discriminator.summary()

discriminator.trainable = False

#initial data for discriminator
noise = np.random.normal(0.5, 0.1, (xtrain.shape[0],seq_length))
real = np.ones((xtrain.shape[0],1))
fake = np.zeros((xtrain.shape[0],1))
xText = np.concatenate((noise,xtrain))
xText = np.expand_dims(xText, axis=2)
yText = np.concatenate((fake,real))

generator.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#train the discriminator
discriminator.fit(xText,yText, epochs=100, batch_size=128)

for epoch in range(5):
    print("Epoch : %d" %(epoch+1))
    
    Xnoise = np.random.normal(0.5, 0.1, (xtrain.shape[0],seq_length,1))
    generator.fit(Xnoise, Xtrain, epochs=1)
    
    gen_txts = np.zeros((64,53))
    for i in range(64):
        noise = np.random.normal(0.5, 0.1, (1,seq_length,))
        noise = np.expand_dims(noise, axis=2)
        gen_txts[i] = generator.predict(noise)
    #gen_txts = np.expand_dims(gen_txts, axis=2)
    for i in range(64):
        result = discriminator.predict(gen_txts[i])
        result = result.round().astype(int)
        if result == 1:
            generated = gen_texts*3342
            generated = np.round(generated)
            generated.astype(int)

            texts = list()
            index = generated.ravel()
            for j in range(53):
                texts.append(int2word[index[j]])
            print(texts)

