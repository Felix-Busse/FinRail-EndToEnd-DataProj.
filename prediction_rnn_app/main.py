from fastapi import FastAPI
import finrail_rnn_model
import os
from sqlalchemy import create_engine
import sys
import tensorflow as tf

# Add finrail_rnn_model module to access rnn models for predictions
modules_path = os.path.join(os.getcwd(), 'data_collect_app')
if modules_path not in sys.path:
    sys.path.append(modules_path)

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
@app.get('/predict/commuter/')
async def prediction():
    '''
    '''
    # Load trained model for this purpose
    path_model = os.path.join(os.getcwd(), 'app/finrail_api/rnn_commuter.keras')
    tf.keras.models.load_model(path_model, 
        custom_objects={'Custom_Metric': finrail_rnn_model.Custom_Metric})
    # Load sql query to load time series up to newest date available
    path_sql_query = os.path.join(os.getcwd(), 'app/finrail_api/timeseries_query.txt')
    with open(path_sql_query, 'r') as f:
        sql_query = f.read()
    # Load time series from database
    df = finrail_rnn_model.read_timeseries_from_database(engine=engine, str_query=sql_query)
    return df
