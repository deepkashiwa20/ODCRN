## ECML PKDD 2021 Applied Data Science Paper
### Origin Destination Convolutional Recurrent Network (ODCRN)
### Countrywide Origin-Destination Matrix Prediction and Its Application for COVID-19

### To Zhaonan Wang at 2021/7/1 23:00 UTC+9
* Final codes are at DL-Tank /home/jiang/PycharmProjects/workECMLPKDD21/workODMatrix_paper, already uploaded to this github, see /model. <br>
* Final results are at DL-Tank /home/jiang/PycharmProjects/workECMLPKDD21/save, already uploaded to this github, check /save. <br>

|Model|RMSE|MAE|MAPE|
|:---|---|---|---|
|MonthlyAverage|4528.066|413.558|53.953%|
|CopyLastWeek|5087.257|432.207|55.988%|
|CNN|6472.387|371.192|125.783%|
|ConvLSTM|4656.759|484.184|43.043%|
|STResNet|6611.505|353.473|99.874%|
|PCRN|6685.624|429.217|226.773%|
|STGCN|3599.582|313.727|82.246%|
|DCRNN(bad)|4831.898|481.174|46.549%|
|DCRNN | ? | ? | ?% |
|GraphWaveNet|4822.943|502.095|44.007%|
|GEML| 3743.262 | 371.141 | 45.341% |
|CSTN| 4598.711 | 435.556 | 43.516% |
|MPGCN| ? | ? | ?% |
|ODCRN (w/o DGC)| ? | ? | ?% |
|ODCRN| ? | ? | ?% |
