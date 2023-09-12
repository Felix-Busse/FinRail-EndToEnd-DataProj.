import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pmdarima.arima import AutoARIMA
from pmdarima.pipeline import Pipeline
from pmdarima.preprocessing import BoxCoxEndogTransformer
from sklearn.impute import SimpleImputer

class TrainLengthModel():
    '''Class allows adding time series data and later fit and predict on these using a pmdARIMA Pipeline. 
    '''
    def __init__(self, train_category, day_of_week):
        '''
        '''
        # Properties "train_category" and "day_of_week" to distinguish different instances
        self.train_category = train_category
        self.day_of_week = day_of_week    
        # initialize data with empty DataFrame, so data can be appended
        self.data = pd.DataFrame()
        # Define time series prediction model for this class
        self._model = Pipeline([
            # BoxCox-transformer to obtain equally distributed variance in time series
            ('BoxCox', BoxCoxEndogTransformer()),
            # ARIMA mdoel with m=12 for seasonal pattern in monthly data. Corrected Akaike's creterion
            # to prevent overfitting, when time series is short
            ('ARIMA', AutoARIMA(m=12, information_criterion='aicc', trace=None))
        ])
        # initialize empty attributes for internal use only
        self._median = 0
        self._index_fitted_on = pd.DatetimeIndex([])
        self._index_predicted_on = pd.DatetimeIndex([])
    
    def add_data(self, data):
        '''Method adds time series data to object. No data is added, if duplicate dates in 
        column "departure date" and data already added occur.
        
        Parameters:
        <pandas DataFrame> containing columns "departure_date" (type: "DateTime", month end, "YYYY-MM-DD"),
        "length_m", "train_category" and "day_of_week" as monthly data.
        
        Return:
        self: TrainLengthModel object
        '''
        # Extract "train_category" and "day_of_week" of class instance from passed data
        df_ = (data[(data.train_category == self.train_category) & (data.day_of_week == self.day_of_week)]
               # Cerate DatetimeIndex
               .set_index('departure_date')
               # Set frequence of index to month, used later to detect missing values
               .asfreq('M'))
        # Concatenate to existing data, if no index value overlap, otherwise print message
        try: 
            self.data = pd.concat([self.data, df_[['length_m']]], verify_integrity=True)
        except ValueError:
            # Identify overlapping index values
            overlap = df_.index.isin(self.data.index)
            # Print overlapping time steps
            print('Data send has overlap with stored data. Overlap ignored. Time steps ignored are:\n')
            for i, time_step in enumerate(df_[overlap].index):
                print(f'{time_step:%Y-%m-%d}\n')
            # Add non-overlapping time steps to data storage
            self.data = pd.concat([self.data, df_[~overlap][['length_m']]], 
                              verify_integrity=True)
        return self
    
    def fit(self):
        '''Function will fit model to data stored in class object.
        It will impute missing values with "median" of all data. Then a stepwise process 
        fits the ARIMA model to data.
        
        Return:
        self: TrainLengthModel object
        '''
        # initialize imputer instance, which fills missing values with median of data
        imputer = SimpleImputer(strategy='median')
        # result of impute-process will be time series to which the model is fittet to
        time_series = imputer.fit_transform(self.data).flatten()
        # store median, as it will be used when updating the model (updating only with parameters obtained
        # during model training)
        self._median = imputer.statistics_[0]
        # fit model to time series
        self._model.fit(y=time_series)
        # store DatetimeIndex, as it is needed to locate predictions later
        self._index_fitted_on = self.data.index
        return self
    
    def predict_full_series(self, steps=12, alpha=0.05):
        '''Method returns DataFrame with four columns. Containing training and 
        prediction period values. Note: negative prediction values (for confidence intervals as well) 
        would result in nan, because reverse Box-Cox-transform only works on non-negative values.
        
        Parameters:
        steps <int> number of month to predict in the future.
        alpha <float> number from 0 to 1 defining confidence level for confidence interval of prediction.
        
        Return:
        <pandas DataFrame> DataFrame with columns:
            "departure_date": Month (end of month) to which the data refers.
            "length_m": Average length of all train compositions on specific day of the week in that month
                in training time period and prediction time period
            "conf_int_lower" lower boundaries of confidence interval in prediction time period
            "conf_ing_upper" upper oundaries of confidence interval in prediction time period
        '''
        # make prediction and store predicted values and confidence interval boundaries
        time_series_prediction, conf_intervals = self._model.predict(
            steps, return_conf_int=True, alpha=alpha
        )
        # Create new DatetimeIndex, which contains the period to which the model was fittet and
        # the period of predicted values
        self._index_predicted_on = pd.date_range(self._index_fitted_on[-1] + self._index_fitted_on.freq, 
                                                 periods=steps, freq='M')
        time_series_index = self._index_fitted_on.union(self._index_predicted_on)
        # Load values of period, the model was fitted to in a numpy array
        time_series = self.data.loc[self._index_fitted_on].length_m.values
        # Add the predicted values to this time series
        time_series = np.concatenate((time_series, time_series_prediction))
        # Create DataFrame to be later returned, fill it with time series (training period and prediction)
        df_ = pd.DataFrame(time_series, index=time_series_index, columns=['length_m'])
        # Create new columns containing confidence interval boundaries for prediction and zero
        # on training data
        # Fill nan with zero, as these occur due to negative inputs to box-cox-transformation, time series
        # for total composition length is non-negative anyway.
        df_ = (df_
               .assign(
            conf_int_lower=pd.Series(data=np.concatenate(
                (np.full(shape=len(self._index_fitted_on), fill_value=0), conf_intervals[:, 0])),
                index=time_series_index),
            conf_int_upper=pd.Series(data=np.concatenate(
                (np.full(shape=len(self._index_fitted_on), fill_value=0), conf_intervals[:, 1])),
                index=time_series_index))
               # Replace all nan with zero
               .fillna(0)
               # Reset index, as return should be .json-compatible
               .reset_index(names='departure_date')
              )
        return df_
    
    def predict_and_plot(self, steps=12, alpha=0.05):
        '''Method calls .predict_full_series() method and plots resulting time series (training period +
        prediction including confidence intervals.). Note: negative prediction values (for confidence 
        intervals as well) would result in nan, because reverse Box-Cox-transform only works on 
        non-negative values.
        
        Parameters:
        steps <int> number of month to predict in the future.
        alpha <float> number from 0 to 1 defining confidence level for confidence interval of prediction.
        
        Return:
        self: TrainLengthModel object
        '''
        # Call predict method to obtain DataFrame with time series and confidence intervals
        df_ = self.predict_full_series(steps=steps, alpha=alpha)
        # Set DateTimeIndex
        df_ = df_.set_index('departure_date')
        # Set seaborn to default theme
        sns.set_theme()
        # Plot time series including prediction as a line plot
        g = sns.lineplot(data=df_.length_m)
        # Plot confidence interval as filled area between confidence interval boundaries
        g.axes.fill_between(self._index_predicted_on, df_.loc[self._index_predicted_on].conf_int_lower, 
                            df_.loc[self._index_predicted_on].conf_int_upper, alpha=0.4)
        # Create legend containing day of the week plotted and confidence level
        #g.legend([f'{self.day_of_week}', f'{1-alpha:.0%} confidence'], loc='lower right')
        # Plot on y-axis from zero
        g.axes.set_ylim(0)
        # Set labels on x and y-axis to appropriate names
        g.axes.set_xlabel('Date')
        g.axes.set_ylabel('Length / m')
        # Set plot title to train_category
        g.set_title(f'{self.train_category}')
        # Define legend and set it outside of plot with no background
        plt.legend([f'{self.day_of_week}', f'{1-alpha:.0%} confidence'],
            bbox_to_anchor=(1.02, 0.55), loc='upper left', 
            facecolor='white', edgecolor='white')
        return self
    
    def predict(self, steps=12, alpha=0.05):
        '''Method returns DataFrame with four columns and. Containing  prediction period values. 
        Note: negative prediction values (for confidence intervals as well) 
        would result in nan, because reverse Box-Cox-transform only works on non-negative values.
        
        Parameters:
        steps <int> number of month to predict in the future.
        alpha <float> number from 0 to 1 defining confidence level for confidence interval of prediction.
        
        Return:
        <pandas DataFrame> DataFrame with columns:
            "departure_date": Month (end of month) to which the data refers.
            "length_m": Average length of all train compositions on specific day of the week in that month
                in training time period and prediction time period
            "conf_int_lower" lower boundaries of confidence interval in prediction time period
            "conf_ing_upper" upper oundaries of confidence interval in prediction time period
        '''
        # make prediction and store predicted values and confidence interval boundaries
        time_series_prediction, conf_intervals = self._model.predict(steps, return_conf_int=True, 
                                                                     alpha=alpha)
        # Create new DatetimeIndex, which contains the period to which the model was fittet and
        # the period of predicted values
        self._index_predicted_on = pd.date_range(self._index_fitted_on[-1] + self._index_fitted_on.freq, 
                                                 periods=steps, freq='M')
        # Create DataFrame with DatetimeIndex and predicted values, as well as confidence intervals 
        df_ = pd.DataFrame(data=time_series_prediction, index=self._index_predicted_on, 
                           columns=['length_m'])
        df_ = (df_
               .assign(conf_int_lower=pd.Series(conf_intervals[:, 0], index=self._index_predicted_on), 
                       conf_int_upper=pd.Series(conf_intervals[:, 1], index=self._index_predicted_on))
               # Replace all nan with zero
               .fillna(0)
               # Reset index, as return should be .json-compatible
               .reset_index(names='departure_date')
              )
        return df_
    
    def update(self):
        '''Method will update model to time steps in data, that havn't been used for fitting/updating jet.
        
        Return:
        self: TrainLengthModel object
        '''
        # Extract data, that was not jet used for fitting
        new_data = self.data[~self.data.index.isin(self._index_fitted_on)]
        # Recalculate fitting parameters without performing complete AutoARIMA model finding process
        self._model.update(new_data.length_m.values)
        # Store new index
        self._index_fitted_on = self._index_fitted_on.union(new_data.index)
        return self
    
    def ljung_box_p_value(self):
        '''Method returns p-value of Ljung-Box test. Ljung-Box-Test:
        H0: Residuals show no serial correlation. (abrv.)
        H1: There is serial correlation in residuals. (abrv.)
        
        Return:
        <float> p-value of Ljung-Box-test [0-1]
        '''
        table2_IO = StringIO(self._model.summary().tables[2].as_csv())
        df_ = pd.read_csv(table2_IO, header=None)
        return df_.iloc[1, 1]
