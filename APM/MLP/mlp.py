#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 강제 CPU 사용
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# 0. 사용할 패키지 불러오기
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv
import matplotlib.pyplot as plt

import pdb
#get_ipython().run_line_magic('matplotlib', 'inline')

def create_dataset(signal_data, look_back=1):
    dataX, dataY = [], []
    for i in range(len(signal_data)-look_back):
        dataX.append(signal_data[i:(i+look_back), 0])
        dataY.append(signal_data[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

look_back = 50 # 예측하고자 하는 n step


# In[2]:


# 1. 데이터셋 불러오기
apm_cpu_data = read_csv("dataset/1808-12.csv")
apm_cpu_data = apm_cpu_data[apm_cpu_data['mid'] == 288]
apm_cpu_data = apm_cpu_data.drop(columns='mid') 
apm_cpu_data = apm_cpu_data['cpu.usage'].values[::-1]
apm_cpu_data = apm_cpu_data.reshape(-1, 1)
apm_cpu_data.astype('float32')

# 데이터 전처리
scaler = MinMaxScaler(feature_range=(0, 1))
signal_data = scaler.fit_transform(apm_cpu_data)

# 데이터 분리
train = apm_cpu_data[0:len(apm_cpu_data)//2]   # 50%
val = apm_cpu_data[len(train):(len(train)+len(apm_cpu_data)//4)]  # 25%
test = apm_cpu_data[(len(train)+len(val)):]    # 25%

# #print(len(signal_data))
# print(len(train))
# print(len(val))
# print(len(test))


# In[3]:


# 데이터셋 생성
x_train, y_train = create_dataset(train, look_back)
x_val, y_val = create_dataset(val, look_back)
x_test, y_test = create_dataset(test, look_back)

# 데이터셋 전처리
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

x_train = np.squeeze(x_train)
x_val = np.squeeze(x_val)
x_test = np.squeeze(x_test)

pdb.set_trace()
# In[4]:

# 2. 모델 구성하기
model = Sequential()
model.add(Dense(32,inpt_dim=50,activation="relu"))
# model.add(Dropout(0.5)) # 일반적 모델 -> 주석 해제
for i in range(10):
    model.add(Dense(32,activation="relu"))
#     model.add(Dropout(0.5)) # 일반적 모델 -> 주석 해제
model.add(Dense(1))


# In[5]:


# 3. 모델 학습과정 설정하기
model.compile(loss='mean_squared_error', optimizer='adam')


# In[6]:


# 4. 모델 학습시키기
hist = model.fit(x_train, y_train, epochs=1, batch_size=256, 
                 validation_data=(x_val, y_val))


# In[7]:


# 5. 학습과정 살펴보기
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.ylim(0.0, 0.15)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[8]:


# 6. 모델 평가하기
trainScore = model.evaluate(x_train, y_train, verbose=0)
print('Train Score: ', trainScore)
valScore = model.evaluate(x_val, y_val, verbose=0)
print('Validataion Score: ', valScore)
testScore = model.evaluate(x_test, y_test, verbose=0)
print('Test Score: ', testScore)


# In[9]:


# 7. 모델 사용하기
look_ahead = 60
xhat = x_test[0, None]
predictions = np.zeros((look_ahead,1))
for i in range(look_ahead):
    prediction = model.predict(xhat, batch_size=32)
    predictions[i] = prediction
    xhat = np.hstack([xhat[:,1:],prediction])


# In[10]:



plt.figure(figsize=(15,5))
plt.plot(np.arange(look_ahead),predictions,'r',label="pred")
plt.plot(np.arange(look_ahead),y_test[:look_ahead],label="actual")
plt.legend
fig = plt.gcf()
plt.show()
plt.draw()
fig.savefig('mlp.png')


# In[11]:


model.save("mlp.h5")

