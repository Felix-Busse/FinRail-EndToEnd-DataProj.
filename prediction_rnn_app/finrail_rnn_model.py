import datetime as dt
import numpy as np
import os
import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError
from tensorflow import (cast, reduce_sum, size, float32, transpose, concat, 
    expand_dims, TensorSpec)
from tensorflow.keras import models
from tensorflow.data import AUTOTUNE, Dataset
from tensorflow.keras.metrics import Metric
from tensorflow.math import square, sqrt

class Custom_Metric(Metric):
    '''Metric calculating the root mean squared error (RMSE) for a sequence to 
    sequence recurrent neuronal network (RNN) exclusively based on the last 
    predicted vector of a sequence. This is useful in situation, where a 
    sequence to sequence RNN is trained, but for production only the last 
    predicted vector matters. This occurs for example in time series 
    prediction. This metric allows to evaluate the model performance in time 
    series prediction exclusivley on the parts of output that matters for 
    production. Instead the loss of a sequence to sequence model training takes
    all predicted vectors along a sequence into account.
    '''
    def __init__(self, time_series_index=None, **kwarg):
        '''Function hands over kwargs to parent class and initiates two 
        weights, which will hold the sum of squares and the total count of 
        summed numbers.

        Parameters:
            time_series_index <int> If used with a model, that outputs more 
                than one time series, specify index of time series for which
                custom metric value should be calculated
        '''
        super().__init__(**kwarg) # pass kwargs to parent class
        # Index of time series, if multivariate forecast
        self.time_series_index = time_series_index 
        # Initialize weight for sum of squared errors and count of samples
        self.sum_of_squares = self.add_weight('sum_of_squares', 
            initializer='zeros')
        self.sample_count = self.add_weight('sample_count', 
            initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        ''' Function will add to sum_of_squares and sample_count every 
        batch.'''
        if self.time_series_index == None:
            # True, if class is in use for forecasting single time series
            # Add number of data point in batch so count of samples            
            self.sample_count.assign_add(cast(size(y_pred[:, -1, :]), float32))
            # Sum of squares of errors (y_true - y_pred) of last sequence
            self.sum_of_squares.assign_add(
                reduce_sum(square(y_pred[:, -1, :] - y_true[:, -1, :]))
            )
        else:
            # If class is in use for multivariate forecasting, calculate for 
            # selected time series only
            self.sample_count.assign_add(
                cast(size(y_pred[:, -1, :, self.time_series_index]), float32)
            )
            # sum of squares of errors of (y_true - y_pred) of last sequence
            self.sum_of_squares.assign_add(
                reduce_sum(
                    square(
                        y_true[:, -1, :, self.time_series_index] - 
                        y_pred[:, -1, :, self.time_series_index]
                    )
                )
            )
    
    def result(self):
        '''Function will calculate the RMSE at the end of every epoch'''
        return sqrt(self.sum_of_squares / self.sample_count)
                                    
    def reset_state(self):
        '''Function will reset all stateful variables to zero'''
        self.sample_count.assign(0)
        self.sum_of_squares.assign(0)
        
    def get_config(self):
        '''Function will overwrite get_config() of parent class to 
        include "time_series_index"'''
        conf_dict = super().get_config()
        return {**conf_dict, 'time_series_index': self.time_series_index}
        
def timeseries_window(data, seq_length, shift=1, stride=1):
    '''Function takes dataset and returns dataset containing windows with 
    data from input dataset.

    Parameters:
        data <tf.data.Dataset> input dataset
        seq_length <int> defines length of windows in output dataset
        shift <int> defines how many time steps of gap are between two 
            consecutive windows
        stride <int> defines how many time steps are between two consecutive
            output data points
        
    Return:
        <tf.data.Dataset> Dataset containing windows of seq_length based on 
            input dataset data
    '''
    data = data.window(size=seq_length, shift=shift, stride=stride, 
        drop_remainder=True)
    # Flatten nested Dataset structure returned by .window()
    data = data.flat_map(lambda x: x) 
    # Batch of size seq_length will give one window in each batch
    return data.batch(seq_length) 

def timeseries_dataset_seq2seq(data, forecast_length=1, seq_length=7):
    '''Function takes Dataset and returns Dataset with windows suitable to 
    train a sequence to sequence RNN.
    Parameters:
        data <tf.data.Dataset> input dataset
        forecast_length <int> number of time steps to be forecasted into the 
            future
        seq_length <int> length of sequences fed to RNN (number of consecutive
            time steps in one training instance)
    '''
    # First dimension one time step longer than forecast_length, as targets 
    # are generated as well
    data = timeseries_window(data, forecast_length+1)
    # Second dimension consists of windows of size sequence length
    data = timeseries_window(data, seq_length)
    # map to tuple (training instance, target)
    return data.map(lambda x: (x[:, 0, :], x[:, 1:, 0]), 
        num_parallel_calls=AUTOTUNE)

def prepare_training_dataset(df_, column, row_split, forecast_length=14, 
    seq_length=30, batch_size=32, seed=42, reshuffle_each_iteration=True):
    '''Function takes Dataframe and returns tf.data.Dataset Ready to be used 
    for training.

    Parameters:
        df_ <pd.Dataframe> Dataframe with time series data (np.float32) 
            in columns
        column <string> name of column or list of column names in 
            DataFrame to use
        row_split <tuple of two int> defines row index between data is 
            extracted from df_
        forecast_length <int> number of time steps to be forecasted into the 
            future
        seq_length <int> length of sequences fed to RNN 
            (number of consecutive time steps)
        batch_size <int> batch_size of returned Dataset
        seed <int> random seed for shuffling data
        reshuffle_each_iteration <boolean> Defines wheater Dataset is ot be 
            reshuffled after each training epoch

    Return:
        <tf.data.Dataset> ready for .fit() of an sequence to sequence RNN
    '''
    data = Dataset.from_tensor_slices(
        df_[column][row_split[0]:row_split[1]].values
    )
    data = timeseries_dataset_seq2seq(data, forecast_length, seq_length)
    # Cache, so that previous transformation are only performed ones
    data = data.cache() 
    # Define shuffling
    data = data.shuffle(
        500, seed=seed, reshuffle_each_iteration=reshuffle_each_iteration
    )
    # Batch and prefetching definition
    return data.batch(batch_size=batch_size).prefetch(AUTOTUNE)

def read_timeseries_from_database(engine, str_query):
    '''Function reads from database engine and returns a pandas Dataframe 
    containing data.

    Parameters:
        engine <sqlalchemy engine object> database engine to read from
            str_query <str> query to be executed 
    
    Returns:
        <pandas Dataframe> Dataframe holding the information returned by 
            the query
    '''
    # Use pandas read_sql_query function to execute passed sql query
    with engine.connect() as connection:
        try:
            df_ = pd.read_sql_query(
                text(str_query), connection, index_col='id'
            )
        except ProgrammingError as err: 
            print('Could not read from table "timeseries".')
            err.args = [
                'No data in database. Please wait for database to fill with '
                'data. This may take several hours after installation, as'
                'data_collect_app needs to start from cron deamon first and '
                'than start downloading, processing and storing data.'
            ]
            raise
        except:
            print('Could not read from database.')
            raise
        else: 
            # Reset index in obtained DataFrame and return it
            return df_.reset_index(drop=True)

def tweak_timeseries(df_):
    '''Function updates DateFrame returned from SQL-query and adds 
    one-hot-encoded information about whether next day in series will be a 
    day Mo-Fr, Saturday or Sunday.

    Parameters:
        df_ <pd.DataFrame> DataFrame as in data base table "timeseries" 
            in database "finrail" 

    Returns:
        <pd.DataFrame> DataFrame holding additionally one-hot-encoded 
            information about next day type (3 columns: Mo-Fr, Saturday, 
            Sunday)
    '''
    df_ = (
        df_.astype({
            'commuter': 'float32', # float32, this is default in tensorflow
            'long_distance': 'float32'
        })
        .assign(date=lambda s: s.date.astype('datetime64[ns]'))
        .assign(commuter=lambda s: s.commuter / 1E5) # scale data
        .assign(long_distance=lambda s: s.long_distance / 1E5) # scale data
        # create information about date of next day
        .assign(next_day=lambda s: s.date.shift(-1)) 
    )
    # Fill missing value, created by shift
    df_.iat[-1, -1] = (df_.date.iloc[-1] + dt.timedelta(days=1)) 
    # Translate date of next day to information whether next day is 
    # Mo-Fr ('W'), Sa ('S') or Sunday('H')
    df_ = (
        df_.assign(
            next_day=lambda df_: df_.next_day.case_when([
                (df_.next_day.dt.weekday<5, 'W'),
                (df_.next_day.dt.weekday==5, 'S'),
                (df_.next_day.dt.weekday==6, 'H')
            ])
        )
        # Set to category for efficiency reason
        .astype({'next_day': 'category'}) 
    )
    # Return DataFrame with one-hot-encoded information about next day
    return pd.get_dummies(df_, columns=['next_day'], dtype='float32')

def simple_sequence_pred(model, df_):
    '''Function will predict based on the last 21 time stamps in df_. 
    Prediction will be based on sequence forecasting (in contrast to 
    one-step-ahead forecasting).

    Parameters:
        model <tf.keras.model> model suitable to predict on 
            columns 1 to 4 of df.
        df_ <pd.DataFrame> DataFrame with datetime data in column 0 and 
            data suitable for model to predict on in columns 1 to 4
    Returns:
        <list of dt.date>, <np.ndarray(dim=1)> list is holding date information
            about prediction horizon, np.ndarray is holding predicted values.
    '''
    # predict on data in DataFrame, keep last sequence (the sequence lying 
    # completly in predicted future) and scale back by multiplying with 1E5 
    # to obtain values with proper scale.
    prediction = model.predict(
        df_.iloc[-21:, 1:].to_numpy()[np.newaxis, :, :]
    )[0, -1, :] * 1E5
    # Construct list of date information of prediction horizon
    dates = [
        (df_.iloc[-1, 0] + dt.timedelta(days=i)).date() 
        for i in range(1, len(prediction) + 1)
    ]
    return dates, prediction

def calc_residuals(model, data_test):
    '''Function calculates the residual errors of predictions made by the 
    model. Errors are calculated for every time step in forecast horizon 
    seperately.
    
    Parameters:
        model <tf.keras.model> Model used for predictions
        data_test <tf.data.Dataset> Dataset containing data and targets, to be
            used in model.predict()
    
    Returns:
        <tf.data.Dataset> Dataset containing residual errors for every time 
            step in forecast horizon.
    '''
    # Predict on test dataset and batch result
    prediction = model.predict(data_test, verbose=0)
    prediction = Dataset.from_tensor_slices(prediction).batch(500)
    # Add test dataset to prediction, so target information is contained 
    # in dataset
    prediction = Dataset.zip(prediction, data_test)
    
    # Definition of generator function on prediction dataset, calculating the 
    # difference between prediction and target
    def difference_gen():
        '''Generator calculates difference between prediction and target of 
        "prediction" dataset
        '''
        for pred, (_, timeseries_target) in prediction:
            # use of last predicted sequence exclusivly, as this sequence 
            # consists of time steps lying completly in the future
            yield pred[:, -1, :] - timeseries_target[:, -1, :] 

    # return differences in a dataset
    return Dataset.from_generator(
        difference_gen, output_signature=(TensorSpec(shape=(None, 14), 
        dtype=float32))
    )

def predict_with_errors(model, data_test, df_, bootstrap_size=10, alpha=0.95):
    '''Function will return prediction for next 14 days for time series 
    including confidence intervals.
    
    Parameter:
        model <tf.keras.model> Model used for predictions
        data_test <tf.data.Dataset> Dataset containing data and targets, to be 
            used in model.predict()
        df_ <pd.DataFrame> DataFrame containing two columns: 
            1. 'date' with datetime information and
            2. time series data information
        alpha <float> Size of confidence interval of returned time series
    
    Return:
        <pd.DataFrame> Time series prediction for 14 days, including 
            confidence intervals
    '''
    # Calculate residuals from test data period, in order to use these 
    # residuals for bootstrapping.
    residuals = calc_residuals(model, data_test)
    # Keep only one step ahead errors.
    residuals = residuals.map(lambda x: x[:, 0])
    residuals = residuals.unbatch()
    # Convert residuals to np.ndarray for later use
    residuals = np.array(list(residuals.as_numpy_iterator()))
    
    # Bootstrapping for multistep ahead prediction of non parametric model is 
    # still discussed in science, see:
    # Politis, D.N.; Wu, K.
    # Multi-Step-Ahead Prediction Intervals for Nonparametric Autoregressions 
    # via Bootstrap:
    # Consistency, Debiasing, and Pertinence. 
    # Stats 2023, 6, 839–867. https://doi.org/10.3390/stats6030053

    # For simplicity intervals are determined by bootstrapping from 
    # one-step-ahead predictions.
    # Algorithm:
    # 1. Forecast one step ahead and randomly add a residual from all 
    # one-step-ahead residuals to the value
    # 2. Predict with updated series another one-step-ahead and repeat until 
    # forecast horizon is reached
    # 3. After many repetitions, select confidence boundaries from obtained 
    # predicted time series
    
    # Restrict time series to 21 latest values, as only these will be used
    # by model
    df_ = df_.iloc[-21:].reset_index(drop=True)
    
    # Add columns for error calculation 
    # These added columns are copies of the column holding the time series. 
    # Later these columns will contain the predictions with bootstrapped errors
    # in prediction horizon. 
    # First of these columns will hold one-step-ahead predictions 
    column_dict = {'one_step_ahead': df_.iloc[:, 1]}
    column_dict = column_dict | {
        'error_col_' + str(i): df_.iloc[:, 1] for i in range(bootstrap_size)
    }
    df_ = pd.concat([df_, pd.concat(column_dict, axis=1)], axis=1)
    
    # Create a dataset out of the DataFrame, that can be passed to 
    # model.predict(). Use transposed of the time series columns of DataFrame 
    # to slice the Dataset along columns. Consequently every element in 
    # dataset will be a time series column.
    dataset = Dataset.from_tensor_slices(df_.iloc[-21:, 5:].to_numpy().T)
    # Take columns representing the day type information
    daytypes = df_.iloc[-21:, 2:5].to_numpy().T
    # Add daytype information to every time series in dataset by mapping
    dataset = dataset.map(
        lambda x: concat([expand_dims(x, axis=0), daytypes], axis=0)
    )
    dataset = dataset.map(lambda x: transpose(x)) # Transposing back
    # Batch (Prefetching is not applied, as only one CPU involved)
    dataset = dataset.batch(500)
    
    # Loop over prediction horizon. Do one-step-ahead predictions for multible
    # copies of the time series and concatenate one step ahead predictions to 
    # time series for next prediction iteration. 
    # Add bootstrapped error (randomly choosen from residuals) every iteration. 
    for i in range(14):
        # Predicting, keeping only one-step-ahead predictions
        prediction = model.predict(dataset, verbose=0)[:, -1, 0]
        # Select random residuals and add errors to predictions (but not to the 
        # first, as it will be the prediction without error)
        random_residuals = np.random.choice(residuals, prediction.shape)
        random_residuals[0] = 0 # first time series will be without error
        prediction = prediction + random_residuals
        # Calculate date that fits to one-step-ahead prediction
        one_step_ahead = df_.iloc[-1].date + dt.timedelta(days=1)
        # Calculate the next day to the predicted day, to calculate the 
        # day type information about the next day. These are necessary for the 
        # next iteration of model.predict()
        next_day = one_step_ahead + dt.timedelta(days=1)
        list_day_info = [
            next_day.weekday()==6, 
            next_day.weekday()==5, 
            next_day.weekday()<5
        ]
        # append new predictions with error to Dataframe holding the 
        # time series (saving results)
        df_.loc[len(df_.index)] = (
            [one_step_ahead, 0] + 
            list_day_info + 
            list(prediction*1E5)
        )

        # Appending predictions and information about next day to dataset, to 
        # prepare it for model.predict() of next iteration
        # Multiply list with day information and reshape, in order to 
        # concatenate it to predictions
        day_info = (
            np.array(list_day_info * (bootstrap_size + 1))
            .reshape(bootstrap_size + 1, 3)
        )
        prediction_and_day_info = np.concatenate(
            [prediction.reshape(bootstrap_size+1, -1), 
            day_info], axis=1
        )
        # Create dataset over prediction_and_day_info and zip it with dataset
        dataset_pred = Dataset.from_tensor_slices(prediction_and_day_info)
        dataset = dataset.unbatch()
        dataset = Dataset.zip(dataset, dataset_pred)
        
        # Map and add predictions as values for new day in time series and 
        # neglect oldest day.
        # This moves the 21 day window of time series one step ahead.
        dataset = dataset.map(
            lambda x, y: concat([x[1:, :], expand_dims(y, axis=0)], axis=0)
        )
        # Batch (Prefetching is not applied, as only one CPU involved)
        dataset = dataset.batch(500)
        
    # Select error margins of predictions according to alpha from all 
    # calculated time series
    error_margins = df_.iloc[-14:, 6:].quantile([1-alpha, alpha], axis=1).T
    # Rename columns
    error_margins.rename(
        columns=lambda s: 'err_interval_' + str(round(s, 2)), inplace=True
    )
    # Calculate the 14 day-ahead-prediction (for comparison to one-step-ahead 
    # prediction)
    _, sequence_pred = simple_sequence_pred(model, df_.iloc[:21, :5])
    # Concat one-step-ahead prediction, error margins and 14-days-ahead 
    # prediction to output frame, containing exclusively prediction horizon 
    # (14 days ahead) 
    return pd.concat(
        [
            # Date information column
            df_[['date']][-14:].reset_index(drop=True),
            # Column containing one-step-ahead prediction
            df_[['one_step_ahead']][-14:].reset_index(drop=True),
#           # Columns containing upper and lower limit of error interval
            error_margins.reset_index(drop=True), 
            # Column containing 14 days prediction (multistep prediction)
            pd.DataFrame(sequence_pred, columns=['14_days_ahead'])
        ], axis=1
    )

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
    custom_objects={'Custom_Metric': Custom_Metric}):
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
