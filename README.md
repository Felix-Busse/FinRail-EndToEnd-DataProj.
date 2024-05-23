# Fin_Rail_analysis
 Using open rail data of Finland. This repository aims to collect amazing applications based on rich open data from finnish rail.

## Applications
### Finrail Timeseries Prediction
End-to-End Machine Learning Project

Predicting the workload of finnish rail services in the upcomming days. Train composition data of recent years is aggregated into **daily timeseries**, representing the workload
of the rail system. A recurrent **neuronal network** is trained on these to predict the workload for the upcoming days.

Details of data inspection, neuronal network training and analysis of results can be found in [Jupyter-Notebook](Train_and_Analyse_RNN.ipynb).

The application includes three containerized services. Check out **commented python code** of these for details:
|Name|Purpose|Code link|
|:------|:------|:------|
|my_sql_database|Stores data [ER shema](ER_shema.png)|[Docker-compose](docker-compose.yaml)|
|data_collect_app|Downloads new data to database, if available|[main.py](data_collect_app/main.py)  [Module](data_collect_app/finrail_db.py)  [Dockerfile](data_collect_app/Dockerfile)|
|prediction_rnn_app|Provides APIs with forecast of timeseries|[main.py](prediction_rnn_app/main.py)  [Module](prediction_rnn_app/finrail_rnn_model.py)  [Dockerfile](prediction_rnn_app/Dockerfile)|

## How to use
To locally run the applications, clone this repository and run `docker compose up`.
Make sure to have a docker runtime installed. 

Be aware that this will download approx. 1 GB of data from the APIs of [Digitraffic](https://www.digitraffic.fi/en/) to set up the database.

## Contribute
Possible contributions may include further inspection of data in the database of "Finrail Timeseries Prediction", which includes more data than used by the app. Alternatively
you can use data from other APIs, as up to now only train composition data is used, check out data source APIs [swagger page](https://rata.digitraffic.fi/swagger/). 

You can contribute:
 |Steps|
 |:----------------------|
 |- Fork this repository|
 |- Clone your repository|
 |- Develop new containerized applications|
 |- Commit and push|
 |- After testing: Create Pull request|
 |- Wait for merge|

