from fastapi import FastAPI
import finrail_rnn_model
import numpy as np
import os
import re
from sqlalchemy import create_engine
import sys
from tensorflow.keras import models

# Load environment variable that holds installing directory in container
app_dir = os.environ['APP_DIR']
#app_dir = 

# Load environment variables with informations for data base connection
db_name = os.environ['DB_NAME']
db_usr = os.environ['DB_USER']
db_psw = os.environ['DB_PSW']
db_server = os.environ['DB_SERVER']

# Create database engine with help of loaded environment varialbles
engine = create_engine(f'mysql+mysqlconnector://{db_usr}:{db_psw}@{db_server}/{db_name}')

# Create FastAPI instance
app = FastAPI()

# Define endpoint for delivery of time series prediction of commuter services
@app.get('/prediction/commuter/')
async def prediction_commuter():
    '''Function responses with JSON format answer, containing dates and values of 14-day prediction
    of time series "commuter". Prediction is obtained by multistep-ahead-prediction (in contrast to
    one-step-head-prediction).
    Additional information is given in JSON answer. See swagger documentation for details
    '''
    # Load trained model for this purpose
    path_model = os.path.join(os.getcwd(), f'{app_dir}rnn_commuter.keras')
    model = models.load_model(path_model, 
        custom_objects={'Custom_Metric': finrail_rnn_model.Custom_Metric})
    # Load sql query to load time series up to newest date available
    path_sql_query = os.path.join(os.getcwd(), f'{app_dir}timeseries_query.txt')
    with open(path_sql_query, 'r') as f:
        sql_query = f.read()
    # Load time series from database
    df = finrail_rnn_model.read_timeseries_from_database(engine=engine, str_query=sql_query)
    # Use tweak-function to process DataFrame and add one-hot-encodings, keep only last 21 days.
    df = finrail_rnn_model.tweak_timeseries(df).iloc[-21:, :]
    # Predict next 14 days of time series
    dates, prediction = finrail_rnn_model.simple_sequence_pred(model, 
        df[['date', 'commuter', 'next_day_H', 'next_day_S','next_day_W']])
    # Assembling response
    days = [{'date': str(date), 
            'value': str(val)} for i, (date, val) in enumerate(zip(dates, prediction))]
    no_days = len(prediction)
    response = {'prediction': {'name': 'commuter', 'day_count': no_days,
                                    'start_day': str(dates[0]),
                                    'end_day': str(dates[-1]),
                                    'prediction_type': 'multistep-ahead',
                                    'error_provided': False,
                                    'days': days}}
    return response

# Define endpoint for delivery of time series prediction of long_distance services
@app.get('/prediction/long_distance/')
async def prediction_long_distance():
    '''Function responses with JSON format answer, containing dates and values of 14-day prediction
    of time series "long_distance". Prediction is obtained by multistep-ahead-prediction (in contrast to
    one-step-head-prediction).
    Additional information is given in JSON answer. See swagger documentation for details
    '''
    # Load trained model for this purpose
    path_model = os.path.join(os.getcwd(), f'{app_dir}rnn_long_distance.keras')
    model = models.load_model(path_model, 
        custom_objects={'Custom_Metric': finrail_rnn_model.Custom_Metric})
    # Load sql query to load time series up to newest date available
    path_sql_query = os.path.join(os.getcwd(), f'{app_dir}timeseries_query.txt')
    with open(path_sql_query, 'r') as f:
        sql_query = f.read()
    # Load time series from database
    df = finrail_rnn_model.read_timeseries_from_database(engine=engine, str_query=sql_query)
    # Use tweak-function to process DataFrame and add one-hot-encodings, keep only last 21 days.
    df = finrail_rnn_model.tweak_timeseries(df).iloc[-21:, :]
    # Predict next 14 days of time series
    dates, prediction = finrail_rnn_model.simple_sequence_pred(model, 
        df[['date', 'long_distance', 'next_day_H', 'next_day_S','next_day_W']])
    # Assembling response
    days = [{'date': str(date), 
            'value': str(val)} for i, (date, val) in enumerate(zip(dates, prediction))]
    no_days = len(prediction)
    response = {'prediction': {'name': 'long_distance', 'day_count': no_days,
                                    'start_day': str(dates[0]),
                                    'end_day': str(dates[-1]),
                                    'prediction_type': 'multistep-ahead',
                                    'error_provided': False,
                                    'days': days}}
    return response

# Define endpoint for delivery of time series prediction of long_distance services with errors
@app.get('/prediction/long_distance_w_error/')
async def prediction_long_distance():
    '''Function responses with JSON format answer, containing dates and values of 14-day prediction
    of time series "long_distance". Prediction is obtained by one-step-ahead-prediction (in contrast to
    multistep-head-prediction). Bootstrapping of residuals is used to provide error margins as well.
    Additional information is given in JSON answer. See swagger documentation for details
    '''
    # Load trained model for this purpose
    path_model = os.path.join(os.getcwd(), f'{app_dir}rnn_long_distance.keras')
    model = models.load_model(path_model, 
        custom_objects={'Custom_Metric': finrail_rnn_model.Custom_Metric})
    # Load sql query to load time series up to newest date available
    path_sql_query = os.path.join(os.getcwd(), f'{app_dir}timeseries_query.txt')
    with open(path_sql_query, 'r') as f:
        sql_query = f.read()
    # Load time series from database
    df = finrail_rnn_model.read_timeseries_from_database(engine=engine, str_query=sql_query)
    # Use tweak-function to process DataFrame and add one-hot-encodings, keep only last 21 days.
    df = finrail_rnn_model.tweak_timeseries(df)
    # Set up test data set, as it is needed to calculate residuals, which are needed for bootstrapping
    data_test = finrail_rnn_model.prepare_training_dataset(df, ['long_distance', 'next_day_H', 
        'next_day_S', 'next_day_W'], (2942, None), batch_size=500, 
        reshuffle_each_iteration=False, seq_length=21)
    # Predict next 14 days of time series
    df_pred = finrail_rnn_model.predict_with_errors(model, data_test, df[['date', 'long_distance', 
        'next_day_H', 'next_day_S', 'next_day_W']],
        bootstrap_size=100)
    # Assembling response
    days = [{'date': str(date.date()), 
            'value': str(val),
            'error_lower_limit': str(lower_lim),
            'error_upper_limit': str(upper_lim)} for i, (date, val, lower_lim, upper_lim) in 
            enumerate(zip(df_pred.date, df_pred.one_step_ahead, df_pred.iloc[:, 2], df_pred.iloc[:, 3]))]
    no_days = len(df_pred.one_step_ahead)
    response = {'prediction': {'name': 'long_distance', 'day_count': no_days,
                                    'start_day': str(df_pred.date.iloc[0].date()),
                                    'end_day': str(df_pred.date.iloc[-1].date()),
                                    'prediction_type': 'one-step-ahead',
                                    'error_provided': True,
                                    'days': days}}
    return response

def load_model(model_name, dir=f'{app_dir}'):
    '''
    '''
    os.path.join([dir, model_name])