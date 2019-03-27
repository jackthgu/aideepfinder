from abc import ABC, abstractmethod
import pdb
import numpy as np

def create_dataset(signal_data, look_back=1):
    dataX, dataY = [], []
    for i in range(len(signal_data)-look_back):
        dataX.append(signal_data[i:(i+look_back), 0]) 
        dataY.append(signal_data[i + look_back, 0]) 
    return np.array(dataX), np.array(dataY)


class basic_resource:
    info=dict()
    logday=0
    loghour=0
    logmon=0
    logtime=0
    logyear=0
    def init_arr(self,_obj):
        pass

class data_manager:
    def __init__(self):
        return

class cpu_resource(basic_resource):
    @abstractmethod
    def init_arr(self,_obj):
        self.info=_obj['_source']['cpu']
        self.logday=_obj['_source']['logday']
        self.loghour=_obj['_source']['loghour']
        self.logmon=_obj['_source']['logmon']
        self.logtime=_obj['_source']['logtime']
        self.logyear=_obj['_source']['logyear']




class arr_cpu:
    obj_train_arr=[]
    obj_pred_arr=[]
    ndarr_train=[]
    ndarr_pred=[]
    def init_train_arr(self,_source):
        for obj in _source['hits']['hits']:
            cpu_obj=cpu_resource()
            cpu_obj.init_arr(obj)
            self.obj_train_arr.append(cpu_obj)


    def init_pred_arr(self,_source):
        for obj in _source['hits']['hits']:
            cpu_obj=cpu_resource()
            cpu_obj.init_arr(obj)
            self.obj_pred_arr.append(cpu_obj)


    def __init__(self):
        return

    def to_train_data(self,dim=1440):
        #self.input_data=np.ndarray(shape=(dim,),dtype=int)
        self.ndarr_train=[]
        for obj in self.obj_train_arr:
            self.ndarr_train.append(obj.info['usage'])

        self.ndarr_train = np.asarray(self.ndarr_train)
        self.ndarr_train=np.reshape(self.ndarr_train,(self.ndarr_train.shape[0],1))

        self.x_train, self.y_train = create_dataset(self.ndarr_train, dim)
        self.x_train = np.reshape(self.x_train, (self.x_train.shape[0], self.x_train.shape[1], 1))
        self.x_train = np.squeeze(self.x_train)

    def to_predict_data(self,dim=1440):
        self.ndarr_pred=[]
        for obj in self.obj_pred_arr:
            self.ndarr_pred.append(obj.info['usage'])

        self.ndarr_pred = np.asarray(self.ndarr_pred)
        self.ndarr_pred=np.reshape(self.ndarr_pred,(self.ndarr_pred.shape[0],1))

        self.x_pred, self.y_pred = create_dataset(self.ndarr_pred, dim)
        self.x_pred = np.reshape(self.x_pred, (self.x_pred.shape[0], self.x_pred.shape[1], 1))
        self.x_pred = np.squeeze(self.x_pred)
        
