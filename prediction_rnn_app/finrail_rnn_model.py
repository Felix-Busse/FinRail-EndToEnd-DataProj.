import datetime as dt
import numpy as np
import pandas as pd
from sqlalchemy import text
#import tensorflow as tf
from tensorflow import cast, reduce_sum, size, float32
from tensorflow.data import AUTOTUNE, Dataset
from tensorflow.keras.metrics import Metric
from tensorflow.math import square, sqrt

class Custom_Metric(Metric):
    '''Metric calculating the root mean squared error (RMSE) for a sequence to sequence recurrent
    neuronal network (RNN) exclusively based on the last predicted vector of a sequence. 
    This is useful in situation, where a sequence to sequence RNN is trained, but for production 
    only the last predicted vector matters. This occurs for example in time series prediction.
    This metric allows to evaluate the model performance in time series prediction exclusivley on
    the parts of output that matters for production. Instead the loss of a sequence to sequence 
    model training takes all predicted vectors along a sequence into account.
    
    '''
    def __init__(self, time_series_index=None, **kwarg):
        '''Function hands over kwargs to parent class and initiates two weights, which will 
        hold the sum of squares and the total count of summed numbers.
        Parameters:
            time_series_index <int> If used with a model, that outputs more than one time series, 
            specify index of time series for which custom metric value should be calculated
        '''
        super().__init__(**kwarg) # pass kwargs to parent class
        self.time_series_index = time_series_index #index of time series if multivariate forecast
        self.sum_of_squares = self.add_weight('sum_of_squares', initializer='zeros')
        self.sample_count = self.add_weight('sample_count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        ''' Function will add to sum_of_squares and sample_counts every batch.'''
        if self.time_series_index == None:
            # True, if class is in use for forecasting single time series
            # sum up how many data point in batch will be summed            
            self.sample_count.assign_add(cast(size(y_pred[:, -1, :]), float32))
            # sum of squares of difference of y_true and y_pred on last sequence
            self.sum_of_squares.assign_add(reduce_sum(
                square(y_pred[:, -1, :] - y_true[:, -1, :]))
            )
        else:
            # If class is in use for multivariate forecasting, calculate for selected time series only
            # sum up how many data point in batch will be summed
            self.sample_count.assign_add(cast(
                size(y_pred[:, -1, :, self.time_series_index]), float32)
            )
            # sum of squares of difference of y_true and y_pred on last sequence
            self.sum_of_squares.assign_add(reduce_sum(square(
                y_true[:, -1, :, self.time_series_index] - y_pred[:, -1, :, self.time_series_index]
            )))
    
    def result(self):
        '''Function will calculate the RMSE at the end of every epoch'''
        return sqrt(self.sum_of_squares / self.sample_count)
                                    
    def reset_state(self):
        '''Function will reset all stateful variables to zero'''
        self.sample_count.assign(0)
        self.sum_of_squares.assign(0)
        
    def get_config(self):
        '''Function will overwrite get_config() of parent class to include "time_series_index"'''
        conf_dict = super().get_config()
        return {**conf_dict, 'time_series_index': self.time_series_index}
        
def timeseries_window(data, seq_length, shift=1, stride=1):
    '''Function takes dataset and returns dataset containing windows with data from input dataset.
    Parameters:
        data <tf.data.Dataset> input dataset
        seq_length <int> defines length of windows in output dataset
        shift <int> defines how many time steps of gap are between two consecutive windows
        stride <int> defines how many time steps are between two consecutive output data points
        
    Return:
        <tf.data.Dataset> Dataset containing windows of seq_length based on input dataset data
    '''
    data = data.window(size=seq_length, shift=shift, stride=stride, drop_remainder=True)
    data = data.flat_map(lambda x: x) # flatten nested Dataset structure returned by .window()
    return data.batch(seq_length) # batch of size seq_length will give one window in each batch

def timeseries_dataset_seq2seq(data, forecast_length=1, seq_length=7):
    '''Function takes Dataset and returns Dataset with windows suitable to train a 
    sequence to sequence RNN
    Parameters:
        data <tf.data.Dataset> input dataset
        forecast_length <int> number of time steps to be forecasted into the future
        seq_length <int> length of sequences fed to RNN (number of consecutive time steps 
        in one training instance)
    '''
    data = timeseries_window(data, forecast_length+1) # First dimension one time step longer than
                                                      # forecast_length, as targets are generated as well
    data = timeseries_window(data, seq_length) # Second dimension consists of windows of size sequence length
    # map to tuple (training instance, target)
    return data.map(lambda x: (x[:, 0, :], x[:, 1:, 0]), 
                    num_parallel_calls=AUTOTUNE)

def prepare_training_dataset(df_, column, row_split, forecast_length=14, seq_length=30, 
                             batch_size=32, seed=42, reshuffle_each_iteration=True):
    '''Function takes Dataframe and returns tf.data.Dataset Ready to be used for training
    Parameters:
        df_ <pd.Dataframe> Dataframe with time series data (np.float32) in columns
        column <string> name of column or list of column names in DataFrame to use
        row_split <tuple of two int> defines row index between data is extracted from df_
        forecast_length <int> number of time steps to be forecasted into the future
        seq_length <int> length of sequences fed to RNN (number of consecutive time steps 
        batch_size <int> batch_size of returned Dataset
        seed <int> random seed for shuffling data
        reshuffle_each_iteration <boolean> Defines wheater Dataset is ot be reshuffled after each
        training epoch
    Return:
        <tf.data.Dataset> ready to feed to .fit() of an sequence to sequence RNN
    '''
    data = Dataset.from_tensor_slices(df_[column][row_split[0]:row_split[1]].values)
    data = timeseries_dataset_seq2seq(data, forecast_length, seq_length)
    data = data.cache() # cache, so that previous transformation are only performed ones
    data = data.shuffle(500, seed=seed, reshuffle_each_iteration=reshuffle_each_iteration)
    return data.batch(batch_size=batch_size).prefetch(AUTOTUNE)

def read_timeseries_from_database(engine, str_query):
    '''Function reads from database engine and returns a pandas Dataframe containing data.
    Parameters:
        engine <sqlalchemy engine object> database engine to read from
        str_query <str> query to be executed 
    
    Returns:
        <pandas Dataframe> Dataframe holding the information returned by the query
    '''
    with engine.connect() as connection:
        df_ = pd.read_sql_query(text(str_query), connection, index_col='id')
    return df_.reset_index(drop=True)

def tweak_timeseries(df_):
    '''Function updates DateFrame returned from SQL-query and adds one-hot-encoded information
    about whether next day in series will be a day Mo-Fr, Saturday or Sunday
    Parameters:
        df_ <pd.DataFrame> DataFrame as in data base table "timeseries" in database "finrail" 
    Returns:
        <pd.DataFrame> DataFrame holding additionally one-hot-encoded information about next day
            type (3 columns: Mo-Fr, Saturday, Sunday)
    '''
    df_ = (df_.astype({'commuter': 'float32',  # use float32, as it is default to tensorflow
                       'long_distance': 'float32'
                      })
            .assign(date=lambda s: s.date.astype('datetime64[ns]'))
            .assign(commuter=lambda s: s.commuter / 1E5) # scale data
            .assign(long_distance=lambda s: s.long_distance / 1E5) # scale data
            .assign(next_day=lambda s: s.date.shift(-1)) # create information about date of next day
    )
    df_.iat[-1, -1] = (df_.date.iloc[-1]+dt.timedelta(days=1)) # Fill missing value, created by shift
    # Translate date of next day to information whether next day is Mo-Fr ('W'), Sa ('S') or Sunday('H')
    df_ = (df_.assign(next_day=lambda df_: df_.next_day.case_when([(df_.next_day.dt.weekday<5, 'W'), 
                                                                   (df_.next_day.dt.weekday==5, 'S'),
                                                                   (df_.next_day.dt.weekday==6, 'H')]))
           .astype({'next_day': 'category'}) # set to category for efficiency reason
    )
    # Return DataFrame with one-hot-encoded information about next day
    return pd.get_dummies(df_, columns=['next_day'], dtype='float32')

def simple_sequence_pred(model, df_):
    '''Function will predict based on the last 21 time stamps in df_. Prediction will be based
    on sequence forecasting (in contrast to one-step-ahead forecasting).
    Parameters:
        model <tf.keras.model> model suitable to predict on columns 1 to 4 of df.
        df_ <pd.DataFrame> DataFrame with datetime data in column 0 and data suitable for model 
            to predict on in columns 1 to 4
    Returns:
        <list, np.ndarray(dim=1)> list is holding date information about prediction horizon,
            np.ndarray is holding predicted values.
    '''
    # Idea: assert statement about df_ has more than 21 entries
    # predict on data in DataFrame, keep last sequence (the sequence lying completly in predicted
    # future) and multiply with 1E5 to obtain values with proper scale.
    prediction = model.predict(df_.iloc[-21:, 1:].to_numpy()[np.newaxis, :, :])[0, -1, :] * 1E5
    # Construct list of date information of prediction horizon
    dates = [(df_.iloc[-1, 0] + dt.timedelta(days=i)).date() for i in range(1, len(prediction) + 1)]
    return dates, prediction