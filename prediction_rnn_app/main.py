from fastapi import FastAPI
import numpy as np
import os
import re
from sqlalchemy import create_engine
from sqlalchemy.exc import ProgrammingError 
import sys
from tensorflow.keras import models

import finrail_rnn_model # own module

# Load environment variables with informations for data base connection
db_name = os.environ['DB_NAME']
db_usr = os.environ['DB_USER']
db_psw = os.environ['DB_PSW']
db_server = os.environ['DB_SERVER']

# Create database engine with help of loaded environment varialbles
engine = create_engine(f'mysql+mysqlconnector://{db_usr}:{db_psw}@' +
    f'{db_server}/{db_name}')

# Load environment variable that holds installing directory in container
app_dir = os.environ['APP_DIR']
# Process string and cut it to parts, to work with file path manipulation
app_dir = re.sub('^/|/$', '', app_dir)
app_dir_parts = re.split(os.path.sep, app_dir)
# Add leading path seperator, as this is absolute path
app_dir_parts = [os.path.sep] + app_dir_parts

def load_from_subdirectory(filename, sub_dir_path_list=[]):
    '''Function reads back file passed to function. Subdirectory may be 
    specified.

    Parameters:
        filename <str> Name of file to read from.
        sub_dir_path_list <list of str> List with strings defining a 
            subdirectory from where to read file.

    Returns:
        <str> Content of file.
    '''
    # Concatenate lists to represent complete path to file and join list to 
    # final file path
    path_list = sub_dir_path_list + [filename]
    path = os.path.join(*path_list)
    # Do the reading from 
    try:
        with open(path, 'r') as f:
            return f.read()
    except:
        raise IOError('Could not load file from subdirectory.')

def load_tensorflow_model(filename, sub_dir_path_list=[], 
    custom_objects={'Custom_Metric': finrail_rnn_model.Custom_Metric}):
    '''Function loads tensorflow.keras model. Custom objects of model and 
    subdirectory file path may be specified.

    Parameters:
        filename <str> Name of model file
        sub_dir_path_list <list of str> List with strings defining a
            subdirectory from where to read model.
        custom_objects <dict> Dictionary 'key': Class passed to 
            tf.keras.models.load_model() to define custom objects in model 
            definition
    
    Returns:
        <tf.keras.model> Model loaded from file.
    '''
    # Concatenate lists to represent complete path to model and join list to 
    # final file path
    path_list = [os.getcwd()] + sub_dir_path_list + [filename]
    path = os.path.join(*path_list)
    try:
        return models.load_model(path, custom_objects=custom_objects)
    except:
        raise IOError('Could not load tf.keras model.')

# Create FastAPI instance
app = FastAPI()

# Define endpoint for delivery of time series prediction of commuter services
@app.get('/prediction/commuter/')
async def prediction_commuter():
    '''Function responses with JSON format answer, containing dates and values
    of 14-day prediction of time series "commuter". Prediction is obtained by 
    multistep-ahead-prediction (in contrast to one-step-head-prediction).
    Additional information is given in JSON answer. See swagger documentation
    for details.
    '''
    # Load trained model for this purpose
    try:
        model = load_tensorflow_model('rnn_commuter.keras', app_dir_parts)
    except IOError as err:
        # Print error to terminal
        print(*err.args) 
        # Display error in response of API
        return {'Error': err.args} 
    # Load sql query to load time series up to newest date available
    try:
        sql_query = load_from_subdirectory(
            'timeseries_query.txt', app_dir_parts
        )
    except IOError as err:
        # Print error to terminal
        print(*err.args) 
        # Display error message in response of API
        return {'Error': err.args}
    # Load time series from database
    try:
        df = finrail_rnn_model.read_timeseries_from_database(engine=engine, 
            str_query=sql_query)
    except ProgrammingError as err:
        # Print error to terminal
        print(*err.args)
        # Display error in response of API
        return {'Error': err.args}
    except:
        # Print error to terminal
        print('Could not load time series from database. Check database '
            'server.')
        # Display error in response of API
        return {
            'Error': 'Could not load time series from database. Check '
            'database server.'
        }
    # Check, whether enough data is present to make predictions
    if df.index.size < 21:
        return {
            'Error': 'Not enough time steps in database to start prediction. '
            'Database must hold at least 21 time steps for reasonable '
            'predictions'
        }
    # Use tweak-function to process DataFrame and add one-hot-encodings, keep 
    # only last 21 days.
    df = finrail_rnn_model.tweak_timeseries(df).iloc[-21:, :]
    # Predict next 14 days of time series
    dates, prediction = finrail_rnn_model.simple_sequence_pred(model, 
        df[['date', 'commuter', 'next_day_H', 'next_day_S', 
        'next_day_W']])
    # Assembling response
    days = [
        {
            'date': str(date), 
            'value': str(val)
        }
        for i, (date, val) in enumerate(zip(dates, prediction))
    ]
    no_days = len(prediction)
    return {
        'prediction': {
            'name': 'commuter', 
            'day_count': no_days,
            'start_day': str(dates[0]),
            'end_day': str(dates[-1]),
            'prediction_type': 'multistep-ahead',
            'error_provided': False,
            'days': days
        }   
    }

# Define endpoint for delivery of time series prediction of long_distance
# services
@app.get('/prediction/long_distance/')
async def prediction_long_distance():
    '''Function responses with JSON format answer, containing dates and values 
    of 14-day prediction of time series "long_distance". Prediction is obtained
    by multistep-ahead-prediction (in contrast to one-step-head-prediction).
    Additional information is given in JSON answer. See swagger documentation 
    for details.
    '''
    # Load trained model for this purpose
    try:
        model = load_tensorflow_model('rnn_long_distance.keras', app_dir_parts)
    except IOError as err:
        # Print error to terminal
        print(*err.args) 
        # Display error in response of API
        return {'Error': err.args} 
    # Load sql query to load time series up to newest date available
    try:
        sql_query = load_from_subdirectory(
            'timeseries_query.txt', app_dir_parts
        )
    except IOError as err:
        # Print error to terminal
        print(*err.args) 
        # Display error in response of API
        return {'Error': err.args} 
    # Load time series from database
    try:
        df = finrail_rnn_model.read_timeseries_from_database(engine=engine, 
            str_query=sql_query)
    except ProgrammingError as err:
        # Print error to terminal
        print(*err.args)
        # Display error in response of API
        return {'Error': err.args}
    except:
        # Print error to terminal
        print('Could not load time series from database. Check database '
            'server.')
        # Display error in response of API
        return {
            'Error': 'Could not load time series from database. Check '
            'database server.'
        }
    # Check, whether enough data is present to make predictions
    if df.index.size < 21:
        return {
            'Error': 'Not enough time steps in database to start prediction. '
            'Database must hold at least 21 time steps for reasonable '
            'predictions'
        }
    # Use tweak-function to process DataFrame and add one-hot-encodings, keep 
    # only last 21 days.
    df = finrail_rnn_model.tweak_timeseries(df).iloc[-21:, :]
    # Predict next 14 days of time series
    dates, prediction = finrail_rnn_model.simple_sequence_pred(model, 
        df[['date', 'long_distance', 'next_day_H', 'next_day_S', 
        'next_day_W']])
    # Assembling response
    days = [
        {
            'date': str(date),
            'value': str(val)
        }
        for i, (date, val) in enumerate(zip(dates, prediction))
    ]
    no_days = len(prediction)
    return {
        'prediction': {
            'name': 'long_distance', 
            'day_count': no_days,
            'start_day': str(dates[0]),
            'end_day': str(dates[-1]),
            'prediction_type': 'multistep-ahead',
            'error_provided': False,
            'days': days
        }
    }

# Define endpoint for delivery of time series prediction of long_distance 
# services with errors
@app.get('/prediction/long_distance_w_error/')
async def prediction_long_distance_w_error():
    '''Function responses with JSON format answer, containing dates and values 
    of 14-day prediction of time series "long_distance". Prediction is obtained
    by one-step-ahead-prediction (in contrast to multistep-head-prediction). 
    Bootstrapping of residuals is used to provide error margins as well.
    Additional information is given in JSON answer. See swagger documentation 
    for details.
    '''
    # Load trained model for this purpose
    try:
        model = load_tensorflow_model('rnn_long_distance.keras', app_dir_parts)
    except IOError as err:
        # Print error to terminal
        print(*err.args) 
        # Display error in response of API
        return {'Error': err.args} 
    # Load sql query to load time series up to newest date available
    try:
        sql_query = load_from_subdirectory(
            'timeseries_query.txt', app_dir_parts
        )
    except IOError as err:
        # Print error to terminal
        print(*err.args) 
        # Display error in response of API
        return {'Error': err.args} 
    # Load time series from database
    try:
        df = finrail_rnn_model.read_timeseries_from_database(engine=engine, 
            str_query=sql_query)
    except ProgrammingError as err:
        # Print error to terminal
        print(*err.args)
        # Display error in response of API
        return {'Error': err.args}
    except:
        # Print error to terminal
        print('Could not load time series from database. Check database '
            'server.')
        # Display error in response of API
        return {
            'Error': 'Could not load time series from database. Check '
            'database server.'
        }
    # Check, whether enough data is present to make predictions
    if df.index.size < 21:
        return {
            'Error': 'Not enough time steps in database to start prediction. '
            'Database must hold at least 21 time steps for reasonable '
            'predictions'
        }
    # Use tweak-function to process DataFrame and add one-hot-encodings, keep 
    # only last 21 days.
    df = finrail_rnn_model.tweak_timeseries(df)
    # Set up test data set, as it is needed to calculate residuals, which are 
    # needed for bootstrapping
    data_test = finrail_rnn_model.prepare_training_dataset(df, 
        ['long_distance', 'next_day_H', 'next_day_S', 'next_day_W'], 
        (2942, None), batch_size=500, reshuffle_each_iteration=False, 
        seq_length=21)
    # Predict next 14 days of time series
    df_pred = finrail_rnn_model.predict_with_errors(model, data_test, 
        df[['date', 'long_distance', 'next_day_H', 'next_day_S', 
        'next_day_W']], bootstrap_size=100)
    # Assembling response
    days = [
        {
            'date': str(date.date()), 
            'value': str(val),
            'error_lower_limit': str(lower_lim),
            'error_upper_limit': str(upper_lim)
        } 
        for i, (date, val, lower_lim, upper_lim)
        in enumerate(zip(df_pred.date, df_pred.one_step_ahead, 
        df_pred.iloc[:, 2], df_pred.iloc[:, 3]))
    ]
    no_days = len(df_pred.one_step_ahead)
    return {
        'prediction': {
            'name': 'long_distance', 
            'day_count': no_days,
            'start_day': str(df_pred.date.iloc[0].date()),
            'end_day': str(df_pred.date.iloc[-1].date()),
            'prediction_type': 'one-step-ahead',
            'error_provided': True,
            'days': days
        }
    }

# Define endpoint for delivery of time series prediction of commuter services 
# with errors
@app.get('/prediction/commuter_w_error/')
async def prediction_commuter_w_error():
    '''Function responses with JSON format answer, containing dates and values
    of 14-day prediction of time series "commuter". Prediction is obtained by 
    one-step-ahead-prediction (in contrast to multistep-head-prediction). 
    Bootstrapping of residuals is used to provide error margins as well.
    Additional information is given in JSON answer. See swagger documentation 
    for details.
    '''
    # Load trained model for this purpose
    try:
        model = load_tensorflow_model('rnn_commuter.keras', app_dir_parts)
    except IOError as err:
        # Print error to terminal
        print(*err.args)
        # Display error in response of API
        return {'Error': err.args} 
    # Load sql query to load time series up to newest date available
    try:
        sql_query = load_from_subdirectory(
            'timeseries_query.txt', app_dir_parts
        )
    except IOError as err:
        # Print error to terminal
        print(*err.args) 
        # Display error in response of API
        return {'Error': err.args} 
    # Load time series from database
    try:
        df = finrail_rnn_model.read_timeseries_from_database(engine=engine, 
            str_query=sql_query)
    except ProgrammingError as err:
        print(*err.args) # Print error to terminal
        return {'Error': err.args} # Display error in response of API
    except:
        # Print error to terminal
        print('Could not load time series from database. Check database '
            'server.')
         # Display error in response of API
        return {
            'Error': 'Could not load time series from database. Check '
            'database server.'
        }
    # Check, whether enough data is present to make predictions
    if df.index.size < 21:
        return {
            'Error': 'Not enough time steps in database to start prediction. '
            'Database must hold at least 21 time steps for reasonable '
            'predictions'
        }
    # Use tweak-function to process DataFrame and add one-hot-encodings, keep 
    # only last 21 days.
    df = finrail_rnn_model.tweak_timeseries(df)
    # Set up test data set, as it is needed to calculate residuals, which are 
    # needed for bootstrapping
    data_test = finrail_rnn_model.prepare_training_dataset(df, 
        ['commuter', 'next_day_H', 'next_day_S', 'next_day_W'], (2942, None), 
        batch_size=500, reshuffle_each_iteration=False, seq_length=21)
    # Predict next 14 days of time series
    df_pred = finrail_rnn_model.predict_with_errors(model, data_test, 
        df[['date', 'commuter', 'next_day_H', 'next_day_S', 'next_day_W']],
        bootstrap_size=100)
    # Assembling response
    days = [
        {
            'date': str(date.date()), 
            'value': str(val),
            'error_lower_limit': str(lower_lim),
            'error_upper_limit': str(upper_lim)
        } 
        for i, (date, val, lower_lim, upper_lim) 
        in enumerate(zip(df_pred.date, df_pred.one_step_ahead, 
        df_pred.iloc[:, 2], df_pred.iloc[:, 3]))
    ]
    no_days = len(df_pred.one_step_ahead)
    return {
        'prediction': {
            'name': 'commuter', 
            'day_count': no_days,
            'start_day': str(df_pred.date.iloc[0].date()),
            'end_day': str(df_pred.date.iloc[-1].date()),
            'prediction_type': 'one-step-ahead',
            'error_provided': True,
            'days': days
        }
    }