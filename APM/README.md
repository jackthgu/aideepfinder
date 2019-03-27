# Application Performance Monotor


## Data format
### + Input data format
|MID | TIME | Data |
|:----:|:----:|:----:|
|67|2018-08-01T15:23:57|1.52292811871|
|67|2018-08-01T15:24:57|0.879654824734|
|67|2018-08-01T15:25:57|0.857740581036|
|...|...|...|
|288|2018-12-24T15:28:51|1|
|288|2018-12-24T15:29:51|0|
|288|2018-12-24T15:30:51|0|

### + Output data format

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


## Sample result
The result above is created using a model in XX.py under `RNN` directory.


h3. Requirements
 * python3 based
 * If you do not have the Anaconda , you need to install the Anaconda.
 * "Download":https://www.anaconda.com/download/#linux and bash run the Anaconda official distribution.

h3. Installation
 * Run @conda env create -f conda_requirements.txt@.
