
# coding: utf-8

# In[1]:


# force to use cpu
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# ela import & define code
import sys
import json
import pytz
import certifi
from datetime import datetime
from elasticsearch import Elasticsearch

def get_time(zone):
    if zone in pytz.all_timezones:
        tz = pytz.timezone(zone)
        return datetime.now(tz)
    else:
        print("Wrong Timezone")
        sys.exit(-1)
        
def get_index(es_client):
    indices = es_client.indices.get_alias().keys()
    return sorted(indices)

def chk_index(today_index, index_list, es_client):
    if any(today_index in s for s in index_list):
        # today's index already exists
        return 0
    else:
        # today's index does not exists
        try:
            mapping_setting = json.loads(open("./mapping.json").read())
            # res = es_client.indices.create(index=today_index, body="{")
            res = es_client.indices.create(index=today_index, body=mapping_setting)
            print("[+] Creating Index...", res)
            print()
            return 1
        except Exception as e:
            print(e)
            return -1
        
# keras code
import math
import numpy as np
import pandas as pd
from pandas import read_csv
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def create_dataset(cpu_data, look_back=1):
    dataX, dataY = [], []
    for i in range(len(cpu_data)-look_back):
        dataX.append(cpu_data[i:(i+look_back), 0]) # [1:1+1440]
        dataY.append(cpu_data[(i + look_back):(i + look_back + 1), 0])  # [1]
    return np.array(dataX), np.array(dataY)

look_back =  1440 # 1440 min = 1 day


# In[2]:


# load apm cpu for specific mid  ex:288
apm_cpu_data = read_csv("dataset/1808-12.csv", index_col="logtime")
apm_cpu_data = apm_cpu_data[apm_cpu_data['mid'] == 288]
apm_cpu_data = apm_cpu_data.drop(columns='mid')
# apm_cpu_data.head()


# In[3]:


# extract cpu.usage from apm data
values = apm_cpu_data['cpu.usage'].values
# values # shape (201446, )


# In[4]:


values = values.reshape(-1, 1) # shape (201446, 1)
# values.astype('float32')


# In[5]:


# split data
test = values[-2000:]   # 50%


# In[6]:


# gen dataset
x_test, y_test = create_dataset(test, look_back)


# In[7]:


# len(x_test)


# In[8]:


# pre-process dataset
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
x_test = np.squeeze(x_test)


# In[9]:


model = load_model("./mlp_mae-adam_final.h5")


# In[10]:


# use model to predict
yhat = model.predict(x_test)


# In[11]:


# translate x-axis data to real date format with KST timezone(+09:00)
inputDates = apm_cpu_data.tail(len(x_test)).index
# kstDates = pd.to_datetime(inputDates) + pd.DateOffset(hours=9) # KST
kstDates = pd.to_datetime(inputDates) # UTC, By default

# add extra n minutes for cpu prediction ex)10
# predDates = kstDates.append(pd.DatetimeIndex([kstDates[-1] + pd.DateOffset(minutes=10)]))
predDates = kstDates


# In[12]:


es_client = Elasticsearch("https://r3v4-ela.idap.ai",
                              use_ssl=True, ca_certs=certifi.where())

# Asia/Seoul | Etc/UTC
# zone = input("Enter timezone (list from pytz) : ") or "Etc/UTC"
zone = "Etc/UTC"
local_date = get_time(zone).strftime("%Y-%m-%d")
today_index = "pred-pfmdata-%s" % local_date
index_list = get_index(es_client)

count = 0
while count < 10:
    if chk_index(today_index, index_list, es_client) != -1:
        break
    else:
        count += 1

        if count < 10:
            print("[-] Retry[#%02d] : Checking index list" % count)
        else:
            print("[-] Retry[#10] : Too many error, shutting down...")

j = 0
for i in predDates:
    local_time = i.strftime("%Y-%m-%d"'T'"%H:%M:%S")
    pred = math.ceil(yhat[j][0])
    act = round(y_test[j][0])
    
    if pred == act:
        level = 1
    elif ( (pred - act) < 2 or (pred - act) > -2 ):
        level = 1
    elif ( (act - pred) < 2 or (act - pred) > -2 ):
        level = 1
    else:
        level = 2
        
    if level == 1:
        describe = "pred ≅ actual"
        msg = "current cpu usage is normal"
        
    elif level == 2:
        describe = "pred != actual"
        msg = "current cpu usage is unnormal"
                          
    pred_cpu_data = """
{
    "mid" : %d,
    "pred-time" : "%s",
    "model" : {
        "type" : "mlp",
        "describe" : "다층 퍼셉트론 모델"
    },
    "cpu" : {
        "pred" : %f,
        "actual" : %f
    },
    "warning" : {
        "level" : 1,
        "describe" : "%s",
        "msg" : "%s"
    }
}
    """ % (288, local_time, pred, act, describe, msg)

    try:
        res = es_client.index(today_index, "pred-pfmdata", pred_cpu_data)
        print("[*] Data insert good (time:%s, mid:%d)" % (local_time, 288))

    except Exception as e:
        print("[-] Data insert fail (time:%s, mid:%d)" % (local_time, 288))
        print(e)

