# 엘라스틱서치 데이터 파싱

## 실시간 데이터 파싱 (모델 적용 시 사용 예정)
`pyelasticsearch`의 DSL를 이용 [[구태홍]]

## 배치 데이터 파싱 (모델 개발 시 사용)
파이썬의 [es2csv](https://github.com/taraslayshchuk/es2csv) 패키지를 이용하여 csv 포맷으로 저장
### + usage
```
Usage
-----

 $ es2csv [-h] -q QUERY [-u URL] [-a AUTH] [-i INDEX [INDEX ...]]
          [-D DOC_TYPE [DOC_TYPE ...]] [-t TAGS [TAGS ...]] -o FILE
          [-f FIELDS [FIELDS ...]] [-S FIELDS [FIELDS ...]] [-d DELIMITER]
          [-m INTEGER] [-s INTEGER] [-k] [-r] [-e] [--verify-certs]
          [--ca-certs CA_CERTS] [--client-cert CLIENT_CERT]
          [--client-key CLIENT_KEY] [-v] [--debug]

 Arguments:
  -q, --query QUERY                        Query string in Lucene syntax.               [required]
  -o, --output-file FILE                   CSV file location.                           [required]
  -u, --url URL                            Elasticsearch host URL. Default is http://localhost:9200.
  -a, --auth                               Elasticsearch basic authentication in the form of username:password.
  -i, --index-prefixes INDEX [INDEX ...]   Index name prefix(es). Default is ['logstash-*'].
  -D, --doc-types DOC_TYPE [DOC_TYPE ...]  Document type(s).
  -t, --tags TAGS [TAGS ...]               Query tags.
  -f, --fields FIELDS [FIELDS ...]         List of selected fields in output. Default is ['_all'].
  -S, --sort FIELDS [FIELDS ...]           List of <field>:<direction> pairs to sort on. Default is [].
  -d, --delimiter DELIMITER                Delimiter to use in CSV file. Default is ",".
  -m, --max INTEGER                        Maximum number of results to return. Default is 0.
  -s, --scroll-size INTEGER                Scroll size for each batch of results. Default is 100.
  -k, --kibana-nested                      Format nested fields in Kibana style.
  -r, --raw-query                          Switch query format in the Query DSL.
  -e, --meta-fields                        Add meta-fields in output.
  --verify-certs                           Verify SSL certificates. Default is False.
  --ca-certs CA_CERTS                      Location of CA bundle.
  --client-cert CLIENT_CERT                Location of Client Auth cert.
  --client-key CLIENT_KEY                  Location of Client Cert Key.
  -v, --version                            Show version and exit.
  --debug                                  Debug mode on.
  -h, --help                               show this help message and exit

```


## 데이터 포맷
### es2csv를 이용하여 csv로 저장되는 데이터 포맷
#### + es2csv 실행 예제
```
es2csv -r -q '{ "query": { "bool": { "must": [ { "match_all": {} }, { "range": { "logtime": { "gte": 1532995200000, "lte": 1546300799000, "format": "epoch_millis" } } }, { "bool": { "should": [ { "match_phrase": { "mid": "282" } }, { "match_phrase": { "mid": "67" } }, { "match_phrase": { "mid": "548" } }, { "match_phrase": { "mid": "288" } }, { "match_phrase": { "mid": "372" } }, { "match_phrase": { "mid": "587" } }, { "match_phrase": { "mid": "134" } }, { "match_phrase": { "mid": "298" } }, { "match_phrase": { "mid": "148" } }, { "match_phrase": { "mid": "433" } } ], "minimum_should_match": 1 } } ], "filter": [], "should": [], "must_not": [] } } }' -o file.csv -f mid logtime cpu.usage -i "pfmdata-2018-*" -S mid logtime -s 10000 --debug; head -n 10 file.csv
```


| mid | logtime | cpu.usage |
|:----:|:----:|:----:|
|67|2018-08-01T15:23:57|1.52292811871|
|67|2018-08-01T15:24:57|0.879654824734|
|67|2018-08-01T15:25:57|0.857740581036|
|67|2018-08-01T15:26:57|1.30891144276|
|67|2018-08-01T15:27:57|1.24717497826|
|67|2018-08-01T15:28:57|0.589859426022|
|67|2018-08-01T15:29:57|0.254548490047|
|...|...|...|
|288|2018-12-24T15:23:51|0|
|288|2018-12-24T15:24:51|7|
|288|2018-12-24T15:25:51|0|
|288|2018-12-24T15:26:51|3|
|288|2018-12-24T15:27:51|0|
|288|2018-12-24T15:28:51|1|
|288|2018-12-24T15:29:51|0|
|288|2018-12-24T15:30:51|0|
