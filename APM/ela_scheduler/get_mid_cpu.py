import pdb
from datetime import datetime
from elasticsearch import Elasticsearch

#from MPL.mlpmd import apm_mlp
import sys
sys.path.insert(0, '../')

from MLP.mlpmd import apm_mlp
from source_data import arr_cpu

#for index in es.indices.get('*'):
#    print (index)


class cpuPredict:
    def __init__(self,host,port):
        #es = Elasticsearch([{'host':'35.238.98.6','port':9200}])
        self.es = Elasticsearch([{'host':host,'port':port}])
        self.arrcpu = arr_cpu()

    def process_indices_data(self,dest_index,mid):
        res = self.es.search(index=dest_index,size=1450, body={
            #"_source":["mid","cpu"],
"sort": 
    {
      "logtime": {
        "order": "asc",
        "unmapped_type": "boolean"
      }
    }
  ,
"query": {
    "bool": {
        "must": [
            {
                "match_all": {}
            },
            {
                "match_phrase": {
                    "mid": {
                        "query": mid
                    }
                }
            },
        ],
        "filter": [],
        "should": [],
        "must_not": []
    }
}})
        return res

    def get_indices(mark):
        for index in self.es.indices.get('*'):
            print (index)
        return

    def setTrainTermMid(self,datelist,mid=288):
        res=[]
        for date in datelist:
            res=self.arrcpu.init_train_arr(self.process_indices_data('pfmdata-'+date,mid))
        return res


    def setPredTermMid(self,datelist,mid=288):
        res=[]
        for date in dl:
            res=self.arrcpu.init_pred_arr(self.process_indices_data('pfmdata-'+date,mid))
        return res


    def setPredictTerm(self):
        return


cp = cpuPredict("35.238.98.6",9200)
dl = ['2018-11-01','2018-11-02','2018-11-03','2018-11-04','2018-11-05','2018-11-06','2018-11-07','2018-11-08','2018-11-09','2018-11-10','2018-11-11','2018-11-12','2018-11-13','2018-11-14']
cp.setTrainTermMid(dl)
cp.arrcpu.to_train_data()
am=apm_mlp()
rfn='training_result.h5'
am.set_train_data(cp.arrcpu)
pdb.set_trace()
am.train(rfn)


dl = ['2018-11-05','2018-11-06','2018-11-07','2018-11-08']

#dl = ['2018-11-11','2018-11-12','2018-11-13','2018-11-14','2018-11-15','2018-11-16','2018-11-17','2018-11-18','2018-11-19','2018-11-20','2018-11-21','2018-11-22','2018-11-23','2018-11-24']

pqr=cp.setPredTermMid(dl)

cp.arrcpu.to_predict_data()
fn ='mlp_mae-adam_final.h5'

am=apm_mlp()

am.set_predict_data(cp.arrcpu)
am.loadmodel(fn)

am.prediction(fn)

pdb.set_trace()
'''
for obj in bb['hits']['hits']:
    obj_cpu=obj['_source']['cpu']
    logday=obj['_source']['logday']
    loghour=obj['_source']['loghour']
    logmon=obj['_source']['logmon']
    logtime=obj['_source']['logtime']
    logyear=obj['_source']['logyear']
'''

print(bb)



'''
 '{ "query":
 { "bool":
 { "must":
 [
 { "match_all": {} },
 { "range":
 { "logtime":
 { "gte": 1532995200000, "lte": 1546300799000, "format": "epoch_millis" } } },
 { "bool": { "should": [ { "match_phrase": { "mid": "282" } }, { "match_phrase": { "mid": "67" } }, { "match_phrase": { "mid": "548" } }, { "match_phrase": { "mid": "288" } }, { "match_phrase": { "mid": "372" } }, { "match_phrase": { "mid": "587" } }, { "match_phrase": { "mid": "134" } }, { "match_phrase": { "mid": "298" } }, { "match_phrase": { "mid": "148" } }, { "match_phrase": { "mid": "433" } } ], "minimum_should_match": 1 } } ], "filter": [], "should": [], "must_not": [] } } }'



            "query":
            { "bool":
              { "must":
                [
                    { "match_all": {} },
                    { "range":
                      { "logtime": { "gte": gte, "lte": lte, "format": "epoch_millis" } } },
                    { "bool":
                      { "should": [
                          { "match_phrase": { "mid": mid } } ], "minimum_should_match": 1 } } ],
                "filter": [], "should": [], "must_not": [] } } })
'''
