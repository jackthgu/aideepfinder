import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv
import matplotlib.pyplot as plt

from keras.callbacks import EarlyStopping

from keras.models import load_model

import sys
sys.path.insert(0, '../')

from source_data import create_dataset

import pdb

class apm_mlp:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = Sequential()
        self.model.add(Dense(32,input_dim=1440,activation="relu"))
        for i in range(5):
            self.model.add(Dense(32,activation="relu"))
        self.model.add(Dense(1))
        self.model.compile(loss='mae', optimizer='adam', metrics=['acc'])

    def set_train_data(self,arrcpu):
        self.train_data=arrcpu.ndarr_train
        self.x_train ,self.y_train = arrcpu.x_train,arrcpu.y_train
        self.signal_data = self.scaler.fit_transform(self.train_data)

    def set_predict_data(self,pred_data):
        self.pred_data = pred_data
        self.x_pred ,self.y_pred = pred_data.x_pred,pred_data.y_pred
        #self.signal_data = self.scaler.fit_transform(self.pred_data)


    def train(self,tfn="mlp_mae-adam_final.h5"):
        self.early_stopping = EarlyStopping(patience = 30) 
        self.hist = self.model.fit(self.x_train, self.y_train, epochs=100, batch_size=10, validation_data=(self.x_train, self.y_train), callbacks=[self.early_stopping], verbose=1)
        trainScore = self.model.evaluate(self.x_train, self.y_train, verbose=0)
        print('Train Score: ', trainScore)
        self.model.save(tfn)


    def loadmodel(self,fn):
        self.model = load_model(fn)

    def prediction(self,tfn="mlp_mae-adam_final.h5"):
        """
        self.look_ahead = 1440
        self.xhat = self.x_test[0, None]
        self.predictions = np.zeros((look_ahead,1))
        for i in range(self.look_ahead):
            self.prediction = self.model.predict(xhat, batch_size=32)
            self.predictions[i] = self.prediction
            self.xhat = np.hstack([self.xhat[:,1:],self.prediction])
        """
        self.yhat = self.model.predict(self.x_pred)



    def test(self):
        self.trainScore = self.model.evaluate(x_train, y_train, verbose=0)
        print('Score: ', self.trainScore)
        model.save("mlp_mae-adam_final.h5")

    def print_predict(self):
        predictDates = apm_cpu_data.tail(len(x_test)).index
        kstDates = pd.to_datetime(predictDates) + pd.DateOffset(hours=9)



