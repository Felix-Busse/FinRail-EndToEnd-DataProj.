import datetime as dt
import finrail_db
import os
import re
from requests.exceptions import ConnectionError
from sqlalchemy import create_engine

# Read environment variables needed to connect to database
db_psw = os.environ['DB_PSW']
db_user = os.environ['DB_USER']
db_name = os.environ['DB_NAME']
db_server = os.environ['DB_SERVER']

# Create database string from content of variables
db_str = f'mysql+mysqlconnector://{db_user}:{db_psw}@{db_server}/{db_name}'

# Read environment variable containing file path of app
app_dir = os.environ['APP_DIR']

# Process string and cut it to parts, to work with file path manipulation
app_dir = re.sub('^/|/$', '', app_dir)
app_dir_parts = re.split(os.path.sep, app_dir)
# Add file seperator at begin, as this is absolute path
app_dir_parts = [os.path.sep] + app_dir_parts

# Create database engine and open session
engine = finrail_db.create_tables(db_str)

# Construct file path for file containing sql query
sql_query_parts = app_dir_parts + ['sql_query.txt']
path_sql_query = os.path.join(*sql_query_parts)

# Pass session to update data base with missing data, use today as end date
try:
    finrail_db.add_compositions(
        engine=engine, date_end=dt.date.today(), verbose=1
    )
except ConnectionError as err:
    print('No connection could be established to source API. Check network.')
    print(*err.args)
except Exception as err:
    print(*err.args)

# Update table "timeseries" using data stored in data base
try:
    finrail_db.update_timeseries(
        engine=engine, path_query=path_sql_query
    )
except Exception as err:
    print(*err.args)