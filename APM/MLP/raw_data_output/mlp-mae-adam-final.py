
# coding: utf-8

# In[1]:


# force to use cpu
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import pdb
import numpy as np
import pandas as pd
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

def create_dataset(signal_data, look_back=1):
    dataX, dataY = [], []
    for i in range(len(signal_data)-look_back):
        dataX.append(signal_data[i:(i+look_back), 0]) 
        dataY.append(signal_data[i + look_back, 0]) 
    return np.array(dataX), np.array(dataY)

look_back =  1440 # 1440 min = 1 day


# In[2]:


# load apm cpu for specific mid  ex:288
apm_cpu_data = read_csv("dataset/1808-12.csv", index_col="logtime")
apm_cpu_data = apm_cpu_data[apm_cpu_data['mid'] == 288]
apm_cpu_data = apm_cpu_data.drop(columns='mid')
apm_cpu_data.info()


# In[3]:


# extract cpu.usage from apm data
values = apm_cpu_data['cpu.usage'].values
values = values.reshape(-1, 1)
values.astype('float32')

# pre-process data
scaler = MinMaxScaler(feature_range=(0, 1))
signal_data = scaler.fit_transform(values)

# split data
train = values[0:len(values)//2]   # 50%
val = values[len(train):(len(train)+len(apm_cpu_data)//4)]  # 25%
test = values[(len(train)+len(val)):]    # 25%


# In[4]:


# gen dataset
x_train, y_train = create_dataset(train, look_back)
x_val, y_val = create_dataset(val, look_back)
x_test, y_test = create_dataset(test, look_back)


# In[5]:


# pre-process dataset
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

x_train = np.squeeze(x_train)
x_val = np.squeeze(x_val)
x_test = np.squeeze(x_test)


# In[6]:

# generate train model
model = Sequential()
model.add(Dense(32,input_dim=1440,activation="relu"))
for i in range(5):
    model.add(Dense(32,activation="relu"))
model.add(Dense(1))


# In[7]:


#  set train setting
model.compile(loss='mae', optimizer='adam', metrics=['acc'])


# In[8]:


# train model
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience = 30) 
hist = model.fit(x_train, y_train, epochs=100, batch_size=10, 
                 validation_data=(x_val, y_val), callbacks=[early_stopping], verbose=1)


# In[9]:


# evulate model
trainScore = model.evaluate(x_train, y_train, verbose=0)
print('Train Score: ', trainScore)
valScore = model.evaluate(x_val, y_val, verbose=0)
print('Validataion Score: ', valScore)
testScore = model.evaluate(x_test, y_test, verbose=0)
print('Test Score: ', testScore)


# In[10]:


# save model with weight and bias
model.save("mlp_mae-adam_final.h5")


# In[11]:


# use model to predict
yhat = model.predict(x_test)

pdb.set_trace()


# In[12]:


# translate x-axis data to real date format with KST timezone(+09:00)
predictDates = apm_cpu_data.tail(len(x_test)).index
kstDates = pd.to_datetime(predictDates) + pd.DateOffset(hours=9)


# In[15]:


j = 0
for i in kstDates:
    act = y_test[j]
    pred = round(yhat[j][0], 2)
    diff = round(yhat[j][0] - y_test[j], 2)
    
    if pred < 0:
        pred = 0
    elif pred == 0:
        pred = 0
    
    if diff > 0:
        pm = 1
    elif diff == 0:
        pm = 0
    elif diff < 0:
        pm = -1
    
    
    if act <=  0:
        act = -1
        print ("[XXX] => %s : cpu usage data wasn't collected\n" % i)
    else:
        print ("Time : %s" % i)
        print ("Actual : %s  |  Predict : %s  |  diff : %s  |  pm : %s\n" % (act, pred, diff, pm) )
    
    j += 1

