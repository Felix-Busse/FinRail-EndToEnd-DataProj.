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

# Create string for app description (metadata)
description = '''
***
<br>
## Measures
As a measure for **workload** of rail services, the sum of the length
of all passengers trains provided per day is used. This measure is considered 
appropriate, as it will increase when more trains are running in the network, 
but as well, when trains with more wagons are used. Thus reflecting both 
factors for stress for rail services: 
- **High usage** of rail network infrastructure
- **Small reserves** of rolling stock

As business case of **commuter trains** differs significantly from those of 
**long-distance trains**, these cases are treated separately. Two APIs are
provided for each case. Use these to obtain timeseries prediction with or 
without error margins. <br><br>
## Model
Sequentially trained LSTM <br><br>
## Data
Data of train compositions from [Fintraffic](https://www.digitraffic.fi/en/)
is used and aggregated.
'''

# Create string for app summary (metadata)
summary = '''
Finrail Timeseries Prediction provides 14-day-ahead preditions for workload
of finnish passenger rail services.
'''
# Create dictionary with metadata for documentation side
metadata = {
    'title': 'Finrail Timeseries Prediction',
    'description': description,
    'summary': summary,
    'contact': {
        'name': 'Felix Busse', 
        'url': 'https://www.github.com/Felix-Busse',
        'email': 'f.busse@posteo.de'
    },
    'version': '1.0.0',
    'license_info': {
        'name': 'GNU General Public License',
        'url': 'https://www.gnu.org/licenses/gpl-3.0.en.html'
    }
}

#Create metadata for tags (topics of endpoints in swagger page)
tags_metadata = [
    {
        'name': 'Commuter',
        'description': '''Prediction of total 
            train composition length for next 14 day.'''
    },
    {
        'name': 'Commuter, with error',
        'description': '''Prediction of total train composition length
            for next 14 day. Error margin of prediction is provided.'''
    },    
    {
        'name': 'Long-distance',
        'description': '''Prediction of total train composition length
            for next 14 day.'''
    },
    {
        'name': '''Long-distance, with error''',
        'description': '''Prediction of total train composition length
            for next 14 day. Error margin of prediction is provided.'''
    }
]

# Create FastAPI instance
app = FastAPI(**metadata, openapi_tags=tags_metadata)

# Define endpoint for delivery of time series prediction of commuter services
@app.get('/prediction/commuter/', tags=['Commuter'])
async def prediction_commuter():
    '''<p>Function responses with JSON format answer, containing dates and
    values of 14-day prediction of time series "commuter". <br>Prediction is
    obtained by multistep-ahead-prediction (in contrast to
    one-step-head-prediction).</p><br>
    Structure of JSON response:\n
        {
            'prediction': {
                'name': <str> Name of time series, 
                'day_count': <int> Number of days in prediction horizon,
                'start_day': <str> Date of first predicted time step,
                'end_day': <str> Date of last predicted time step,
                'prediction_type': <str> Tells whether prediction is "multistep-ahead" or "one-step-ahead" type,
                'error_provided': <Bool> Tells whether prediction includes error margins,
                'days': (list with as many entry as predicted time steps) [
                    {
                        'date': <str> Date of predicted time step,
                        'value': <float> Value of predicted time series at this time step
                    }
                ] 
            }
        }
    '''
    # Load trained model for this purpose
    try:
        model = finrail_rnn_model.load_tensorflow_model(
            'rnn_commuter.keras', app_dir_parts
        )
    except IOError as err:
        # Print error to terminal
        print(*err.args) 
        # Display error in response of API
        return {'Error': err.args} 
    # Load sql query to load time series up to newest date available
    try:
        sql_query = finrail_rnn_model.load_from_subdirectory(
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
    dates, prediction = finrail_rnn_model.simple_sequence_pred(
        model, df[['date', 'commuter', 'next_day_H', 'next_day_S', 
        'next_day_W']]
    )
    # Assembling response
    days = [
        {
            'date': str(date), 
            'value': round(float(val), 2)
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

# Define endpoint for delivery of time series prediction of commuter services 
# with errors
@app.get('/prediction/commuter_incl_error/', tags=['Commuter, with error'])
async def prediction_commuter_incl_error(alpha: float=0.95):
    '''<p>Function responses with JSON format answer, containing dates and
    values of 14-day prediction of time series "commuter". <br>Prediction 
    is obtained by multistep-ahead-prediction (in contrast to
    one-step-head-prediction). Errors are provided. </p><br>
    Structure of JSON response:\n
        {
            'prediction': {
                'name': <str> Name of time series, 
                'day_count': <int> Number of days in prediction horizon,
                'start_day': <str> Date of first predicted time step,
                'end_day': <str> Date of last predicted time step,
                'prediction_type': <str> Tells whether prediction is "multistep-ahead" or "one-step-ahead" type,
                'error_provided': <Bool> Tells whether prediction includes error margins,
                'alpha': <float> Values 0 to 1, defines error margins,
                'days': (list with as many entry as predicted time steps) [
                    {
                        'date': <str> Date of predicted time step,
                        'value': <float> Value of predicted time series at this time step
                        'error_lower_limit' <float> Lower limit of error interval, as to parameter alpha
                        'error_upper_limit' <float> Upper limit of error interval, as to parameter alpha
                    }
                ] 
            }
        }
    '''
    # Load trained model for this purpose
    try:
        model = finrail_rnn_model.load_tensorflow_model(
            'rnn_commuter.keras', app_dir_parts
        )
    except IOError as err:
        # Print error to terminal
        print(*err.args)
        # Display error in response of API
        return {'Error': err.args} 
    # Load sql query to load time series up to newest date available
    try:
        sql_query = finrail_rnn_model.load_from_subdirectory(
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
    data_test = finrail_rnn_model.prepare_training_dataset(
        df, ['commuter', 'next_day_H', 'next_day_S', 'next_day_W'], 
        (2942, None), batch_size=500, reshuffle_each_iteration=False, 
        seq_length=21
    )
    # Predict next 14 days of time series
    df_pred = finrail_rnn_model.predict_with_errors(
        model, data_test, 
        df[['date', 'commuter', 'next_day_H', 'next_day_S', 'next_day_W']],
        bootstrap_size=100, alpha=alpha
    )
    # Assembling response
    days = [
        {
            'date': str(date.date()), 
            'value': round(float(val), 2),
            'error_lower_limit': round(float(lower_lim), 2),
            'error_upper_limit': round(float(upper_lim), 2)
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
            'alpha': alpha,
            'days': days
        }
    }

# Define endpoint for delivery of time series prediction of long_distance
# services
@app.get('/prediction/long_distance/', tags=['Long-distance'])
async def prediction_long_distance():
    '''<p>Function responses with JSON format answer, containing dates and
    values of 14-day prediction of time series "long_distance". <br>Prediction 
    is obtained by multistep-ahead-prediction (in contrast to
    one-step-head-prediction).</p><br>
    Structure of JSON response:\n
        {
            'prediction': {
                'name': <str> Name of time series, 
                'day_count': <int> Number of days in prediction horizon,
                'start_day': <str> Date of first predicted time step,
                'end_day': <str> Date of last predicted time step,
                'prediction_type': <str> Tells whether prediction is "multistep-ahead" or "one-step-ahead" type,
                'error_provided': <Bool> Tells whether prediction includes error margins,
                'days': (list with as many entry as predicted time steps) [
                    {
                        'date': <str> Date of predicted time step,
                        'value': <float> Value of predicted time series at this time step
                    }
                ] 
            }
        }
    '''
    # Load trained model for this purpose
    try:
        model = finrail_rnn_model.load_tensorflow_model(
            'rnn_long_distance.keras', app_dir_parts
        )
    except IOError as err:
        # Print error to terminal
        print(*err.args) 
        # Display error in response of API
        return {'Error': err.args} 
    # Load sql query to load time series up to newest date available
    try:
        sql_query = finrail_rnn_model.load_from_subdirectory(
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
    dates, prediction = finrail_rnn_model.simple_sequence_pred(
        model, df[['date', 'long_distance', 'next_day_H', 'next_day_S', 
        'next_day_W']]
    )
    # Assembling response
    days = [
        {
            'date': str(date),
            'value': round(float(val), 2)
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
@app.get(
    '/prediction/long_distance_incl_error/', tags=['Long-distance, with error']
)
async def prediction_long_distance_incl_error(alpha: float=0.95):
    '''<p>Function responses with JSON format answer, containing dates and
    values of 14-day prediction of time series "long_distance". <br>Prediction 
    is obtained by multistep-ahead-prediction (in contrast to
    one-step-head-prediction). Errors are provided. </p><br>
    Structure of JSON response:\n
        {
            'prediction': {
                'name': <str> Name of time series, 
                'day_count': <int> Number of days in prediction horizon,
                'start_day': <str> Date of first predicted time step,
                'end_day': <str> Date of last predicted time step,
                'prediction_type': <str> Tells whether prediction is "multistep-ahead" or "one-step-ahead" type,
                'error_provided': <Bool> Tells whether prediction includes error margins,
                'alpha': <float> Values 0 to 1, defines error margins,
                'days': (list with as many entry as predicted time steps) [
                    {
                        'date': <str> Date of predicted time step,
                        'value': <float> Value of predicted time series at this time step
                        'error_lower_limit' <float> Lower limit of error interval, as to parameter alpha
                        'error_upper_limit' <float> Upper limit of error interval, as to parameter alpha
                    }
                ] 
            }
        }
    '''
    # Load trained model for this purpose
    try:
        model = finrail_rnn_model.load_tensorflow_model(
            'rnn_long_distance.keras', app_dir_parts
        )
    except IOError as err:
        # Print error to terminal
        print(*err.args) 
        # Display error in response of API
        return {'Error': err.args} 
    # Load sql query to load time series up to newest date available
    try:
        sql_query = finrail_rnn_model.load_from_subdirectory(
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
    data_test = finrail_rnn_model.prepare_training_dataset(
        df, ['long_distance', 'next_day_H', 'next_day_S', 'next_day_W'], 
        (2942, None), batch_size=500, reshuffle_each_iteration=False, 
        seq_length=21
    )
    # Predict next 14 days of time series
    df_pred = finrail_rnn_model.predict_with_errors(
        model, data_test, 
        df[['date', 'long_distance', 'next_day_H', 'next_day_S', 
        'next_day_W']], bootstrap_size=100, alpha=alpha
    )
    # Assembling response
    days = [
        {
            'date': str(date.date()), 
            'value': round(float(val), 2),
            'error_lower_limit': round(float(lower_lim), 2),
            'error_upper_limit': round(float(upper_lim), 2)
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
            'alpha': alpha,
            'days': days
        }
    }