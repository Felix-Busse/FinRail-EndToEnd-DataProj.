import joblib
import os
import sys
from fastapi import FastAPI

# Add path of subdirectory containing own modules
modules_path = os.path.join(os.getcwd(), 'modules')
if modules_path not in sys.path:
    sys.path.append(modules_path)

# Load modules with own clases
from train_compositions import TrainLengthModel

# Initiate instance of FastAPI application
app = FastAPI()

# Define subdirectory of trained models and load models from there
models_file_name = 'trained_time_series.pkl'
models_path = os.path.join(os.getcwd(), 'models', models_file_name)
models = joblib.load(models_path)

@app.get('/predict/all')
async def predict_all(steps: int = 12, alpha: float=0.05):
    '''
    '''
    result_dict = {}
    for i, model in enumerate(models):
        result_dict.update({('model_'+str(i)): {
            'train_category': model.train_category,
            'day_of_week': model.day_of_week,
            'prediction': model.predict(steps=steps, alpha=alpha)}
        })
    return result_dict

@app.get('/predict/{train_category}')#/{day_of_week}')
async def predict(train_category: str='Long_distance', day_of_week: str='Monday', 
                  steps: int=12, alpha: float=0.05):
    for i, model in enumerate(models):
        if((model.train_category == train_category)):
            return {
            'train_category': model.train_category,
            'day_of_week': model.day_of_week,
            'prediction': model.predict(steps=steps, alpha=alpha)
            }
