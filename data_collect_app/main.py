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

# Create database string from content of variables
db_str = f'mysql+mysqlconnector://{db_user}:{db_psw}@{db_server}/{db_name}'

# Create database engine and open session
engine = finrail_db.create_tables(db_str)
Session = sessionmaker(bind=engine)
session = Session()

# Pass session to update data base with missing data, use today as end date
finrail_db.add_compositions(s=session, date_end=dt.date.today())
# Update table "timeseries" using data stored in data base
finrail_db.update_timeseries(s=session, engine=engine)