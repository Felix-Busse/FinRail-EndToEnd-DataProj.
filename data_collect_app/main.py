import datetime as dt
import finrail_db
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import sys

# Add path of subdirectory containing own modules
#modules_path = os.path.join(os.getcwd(), 'app', 'cron')
#if modules_path not in sys.path:
#    sys.path.append(modules_path)
engine = finrail_db.create_tables()
Session = sessionmaker(bind=engine)
session = Session()

finrail_db.add_compositions(s=session, date_end=dt.date.today())