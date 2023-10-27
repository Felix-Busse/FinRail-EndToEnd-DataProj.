import joblib
import matplotlib.pyplot as plt
import os
import pandas as pd
import subprocess
import sys
from fastapi import FastAPI

# Add path of subdirectory containing own modules
modules_path = os.path.join(os.getcwd(), 'modules')
if modules_path not in sys.path:
    sys.path.append(modules_path)

# Load own class
from train_compositions import TrainLengthModel

# Initiate instance of FastAPI application
app = FastAPI()

@app.get('/API/prediction')
async def predict_all(steps: int = 12, alpha: float=0.05):
    '''Function returns predictions for all combinations of "train_category" and "day_of_week" available.
    
    Parameters:
    steps <int> (optional): length of prediction period (months). Default = 12
    alpha <float> (optional): confidence level of predictions. Default = 0.05
    '''    
    # Define subdirectory of trained models and load models from there
    models_file_name = 'trained_time_series.pkl'
    models_path = os.path.join(os.getcwd(), 'models', models_file_name)
    models = joblib.load(models_path)
    
    # Create empty dictionary, than iterate over all models available and add model's attributes to dictionary
    result_dict = {} 
    for i, model in enumerate(models):
        result_dict.update({('model_'+str(i)): {
            'train_category': model.train_category,
            'day_of_week': model.day_of_week,
            'prediction': model.predict(steps=steps, alpha=alpha)}
        })
    # Return dictionary with properties (especially predictions) of all models
    return result_dict

@app.get('/API/prediction/{train_category}/{day_of_week}')
async def predict(day_of_week: str, train_category: str, 
                  steps: int=12, alpha: float=0.05):
    '''Function returns predictions for specified combination of "train_category" and "day_of_week". 
    
    Parameters:
    train_category <string>: Either "Commuter" or "Long-distance"
    day_of_week <string>: Day of week, "Monday" to "Sunday"
    steps <int> (optional): length of prediction period (months). Default = 12
    alpha <float> (optional): confidence level of predictions. Default = 0.05
    '''
    # Define subdirectory of trained models and load models from there
    models_file_name = 'trained_time_series.pkl'
    models_path = os.path.join(os.getcwd(), 'models', models_file_name)
    models = joblib.load(models_path)
    
    # Loop through all models untill model with matching attributes found; return defining attributes and prediction
    for i, model in enumerate(models):
        if((model.train_category == train_category) and 
           (model.day_of_week == day_of_week)):
            return {
            'train_category': model.train_category,
            'day_of_week': model.day_of_week,
            'prediction': model.predict(steps=steps, alpha=alpha)
            }

@app.get('/API/prediction/full_series/{train_category}/{day_of_week}')
async def predict_full_series(day_of_week: str, train_category: str, 
                              steps: int=12, alpha: float=0.05):
    '''Function returns training and prediction time period for specified combination of 
    "train_category" and "day_of_week". 
        
    Parameters:
    train_category <string>: Either "Commuter" or "Long-distance"
    day_of_week <string>: Day of week, "Monday" to "Sunday"
    steps <int> (optional): length of prediction period (months). Default = 12
    alpha <float> (optional): confidence level of predictions. Default = 0.05
    '''
    # Define subdirectory of trained models and load models from there
    models_file_name = 'trained_time_series.pkl'
    models_path = os.path.join(os.getcwd(), 'models', models_file_name)
    models = joblib.load(models_path)    
    
    # Loop through all models untill model with matching attributes found; return defining attributes and full time 
    # series
    for i, model in enumerate(models):
        if((model.train_category == train_category) and 
           (model.day_of_week == day_of_week)):
            return {
            'train_category': model.train_category,
            'day_of_week': model.day_of_week,
            'full_series': model.predict_full_series(steps=steps, alpha=alpha)
            }
    
@app.get('/API/plotting/{train_category}/{day_of_week}')
async def predict_and_plot(day_of_week: str, train_category: str,
                           steps: int=12, alpha: float=0.05, 
                           file_name: str='time_series_plotting.png'):
    '''Function will plot full time series (composed of training data period and prediction period) together with
    confidence intervals. Plot will be stored as .png.
    
    Parameters:
    train_category <string>: Either "Commuter" or "Long-distance"
    day_of_week <string>: Day of week, "Monday" to "Sunday"
    steps <int> (optional): length of prediction period (months). Default = 12
    alpha <float> (optional): confidence level of predictions. Default = 0.05
    file_name <str> (optional): file name of stored plot (.png)
    
    Returns:
    <json> with "plot_successful" as bool
    '''
    # Define subdirectory of trained models and load models from there
    models_file_name = 'trained_time_series.pkl'
    models_path = os.path.join(os.getcwd(), 'models', models_file_name)
    models = joblib.load(models_path)
    
    # Initiate variable to measure, whether plot was executed
    success = False
    # Define path and file name to store plot figure
    plot_path = os.path.join(os.getcwd(), 'data', file_name)
    # Remove file in plot_path, if exists
    subprocess.run(['rm', plot_path])
    # Clear platplotlib figure
    plt.clf()
    # Loop through all models untill model with matching attributes found; plot and save to file
    for i, model in enumerate(models):
        if((model.train_category == train_category) and 
           (model.day_of_week == day_of_week)):
            model.predict_and_plot(steps=steps, alpha=alpha)
            plt.savefig(plot_path, bbox_inches="tight")
            success = True   
    # Return whether a plot was succe
    return {'plot_successful': success}            
            
            
            
            
            
            
            
            
