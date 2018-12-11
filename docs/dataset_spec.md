**Sequential Dataset specification**

# Requirements

We need to support datasets of irregular time series with missing values.

# CSV file spec

We can represent collections of many timeseries (or sequences) by defining a CSV file as follows:


```
subj_uid,time,measure_A,measure_B,measure_C
```

Missing values should be denoted with no-content fields (e.g. in csv they'll be written as ',,'.



