
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import os 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint


# In[ ]:


curr_path = os.getcwd()
print(curr_path)
print(os.listdir())


# ## Data Preparation

# In[ ]:


#load data 
sonnets = open(curr_path + "/haikuzou.txt")

#read first 5 lines 
for i in range(0,5): 
    print(sonnets.readline())


# In[ ]:


lines = []
with open(curr_path + "/haikuzou.txt") as sonnets:
    content = sonnets.readlines() 
    for line in content:
        lines.append(line)


# In[ ]:


lines = list(filter(lambda a: a != "\n" and a != "\"", lines)) #get rid of empty lines 
print(len(lines))


# In[ ]:


#join all lines into one big string 
text = ''.join(elem for elem in lines)
text = text.lower()
text = text[0:len(text)//4]
print(len(text))
unique_chars = sorted(list(set(text)))
print(len(unique_chars))


# In[ ]:


#character mapping 
values = np.arange(len(unique_chars))
number_to_char, char_to_number = {}, {} 
for i in range(0, len(values)):
    number_to_char[values[i]] = unique_chars[i]
    char_to_number[unique_chars[i]] = values[i]


# In[ ]:


print(char_to_number)
print(number_to_char)


# ## Building Sequences

# In[ ]:


#building sequences 
x_train, y_train = [], [] #contains numbers 
sequence_len = 100 
for i in range(0, len(text) - sequence_len): 
    sequence = text[i:i+sequence_len] #get every 100 sequences 
    target = text[i+sequence_len] #get the next character in the sequence 
    y_train.append(char_to_number[target])
    temp_list = []
    for j in range(0, len(sequence)): 
        char = sequence[j]
        char_nb_rep = char_to_number[char]
        temp_list.append(char_nb_rep)
    x_train.append(temp_list)
    temp_list = [] #empty list 


# In[ ]:


#transforms data into suitable shape 
x = np.reshape(x_train, (len(x_train), sequence_len, 1))
#normalise x 
x = x/float(len(unique_chars)) #divide by total number of unique chars
print("x shape: ", x.shape)

#Transform y to one-hot encoded vector 
y = np_utils.to_categorical(y_train)
print("y shape: ", y.shape)
print("Example of one-hot encoded vector: \n", y[0])


# ## Building LSTM Model

# In[ ]:


#Create the model 
model = Sequential()
model.add(LSTM(600, input_shape=(x.shape[1], x.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(600, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(600))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary() 


# In[ ]:


checkpoint = ModelCheckpoint("model-{epoch:02d}-{loss:.4f}.hdf5", monitor='loss', verbose=0, save_best_only = False, save_weights_only = False, mode='auto', period = 1)
callbacks_list = [checkpoint]
#model.load_weights("model-43-0.4930.hdf5")
model.fit(x, y, epochs = 100, batch_size = 100, callbacks = callbacks_list)
model.save_weights('model')


# ## Test on training set

# In[ ]:


text = ''.join(elem for elem in lines)
text = text.lower()
sequence2 = text[70000:70100] #choose random sequence as starting point
sequence = []
for i in range(0, len(sequence2)):
    sequence.append(char_to_number[sequence2[i]])


# In[ ]:


poem = [] 
for j in range(0, len(sequence)): 
    char = number_to_char[sequence[j]]
    poem.append(char) #map back numbers to characters 
    
for i in range(0, 1000): #get the next 1000 characters 
    if i % 10 == 0: 
      
    x = np.reshape(sequence, (1, len(sequence), 1))
    #normalise the sequence
    x = x/float(len(unique_chars))
   
    #make the prediction based on current sequence
    prediction = np.argmax(model.predict(x)) #returns an index 
    predicted_char = number_to_char[prediction]
    
    #add new prediction to poem 
    poem.append(predicted_char)
    #define the next sequence 
    sequence.append(prediction)
    sequence = sequence[1:len(sequence)]


# In[ ]:


text = ""
for char in poem: 
    text += char 
print(text)

