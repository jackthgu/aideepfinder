# Facebook Prophet

## Model description
Developed using Facebook prophet (developed by facebook research).
FBProphet은 Facebook이 개발한 시계열 예측모델로, 가법모델 (additive model)을 기반으로 하며 연간, 주간, 일간, 그리고 기타 휴일로 인한 트렌드에 나타나는 비선형적 특징에 피팅하는 모델이다.
### 중요한 파라미터들
`changepoint_prior_scale`: Changepoint의 개수를 조절하는 파라미터.<br>
`periods`: 예측하는 기간

### 참고자료
[Official webpage](https://facebook.github.io/prophet/)

[Blog post](https://research.fb.com/prophet-forecasting-at-scale/)

[Paper](https://research.fb.com/prophet-forecasting-at-scale/)

## APM Application example
### Sample input and output
#### + Input data format
|MID | TIME | Data |
|:----:|:----:|:----:|
|67|2018-08-01T15:23:57|1.52292811871|
|67|2018-08-01T15:24:57|0.879654824734|
|67|2018-08-01T15:25:57|0.857740581036|
|...|...|...|
|288|2018-12-24T15:28:51|1|
|288|2018-12-24T15:29:51|0|
|288|2018-12-24T15:30:51|0|

#### + Output data format

|Time | Prediction | Lower | Upper
|:----:|:----:|:----:|:----:|
|2018-08-01T15:22:57|23.123|20.344|30.112|
|2018-08-01T15:23:57|28.312|24.294|30.672|
|2018-08-01T15:24:57|22.781|19.993|29.992|
|2018-08-01T15:25:57|21.989|17.120|30.122|
|...|...|...|...|
|2018-12-24T15:28:51|35.23|33.451|39.822|
|2018-12-24T15:29:51|37.22|35.392|39.222|
|2018-12-24T15:30:51|33.24|29.310|36.112|

### Sample figure
![png](https://storage.googleapis.com/apmdata/fb_probhet.png)
