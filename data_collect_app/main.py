import datetime as dt
import finrail_db
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Read environment variables needed to connect to database
db_psw = os.environ['DB_PSW']
db_user = os.environ['DB_USER']
db_name = os.environ['DB_NAME']
db_server = os.environ['DB_SERVER']
# Read environment variable containing file path of app
app_path = os.environ['APP_PATH']

# Create database string from content of variables
db_str = f'mysql+mysqlconnector://{db_user}:{db_psw}@{db_server}/{db_name}'

# Create database engine and open session
engine = finrail_db.create_tables(db_str)
Session = sessionmaker(bind=engine)
session = Session()

# Construct file path for file containing test for sql query
path_sql_query = os.path.join(os.getcwd(), app_path, 'sql_query.txt')

# Pass session to update data base with missing data, use today as end date
finrail_db.add_compositions(s=session, date_end=dt.date.today())
# Update table "timeseries" using data stored in data base
finrail_db.update_timeseries(s=session, engine=engine, path_query=path_sql_query)

# Close session at the end
session.close()