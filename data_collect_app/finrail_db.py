import datetime as dt
import re
import requests
from sqlalchemy import (create_engine, Column, Integer, VARCHAR, DATE, DATETIME, ForeignKey, Boolean, 
                        select, func)
from sqlalchemy.orm import declarative_base, relationship, backref

# Create base class for sqlalchemy class decleration
Base = declarative_base()

class Train(Base):
    '''Defines class object for table "trains", see ER shema for details'''
    
    __tablename__ = 'trains'
    
    uid = Column('id', Integer, autoincrement=True, primary_key=True) #id
    train_number = Column('train_number', Integer) # store train number
    dep_date = Column('dep_date', DATE) # store departure date of train
    operator_code = Column('operator_code', VARCHAR(10)) # store operator short code 
    train_cat = Column('train_cat', VARCHAR(19)) # store train category, for example "long-distance"
    train_type = Column('train_type', VARCHAR(3)) # store train type, for example "IC"
    

class Journey_Section(Base):
    '''Defines class object for table "journey_section", see ER shema for details'''
    
    __tablename__ = 'journey_section'
    
    uid = Column('id', Integer, autoincrement=True, primary_key=True) #id
    train_id = Column('train_id', Integer, ForeignKey('trains.id')) # id of corresponding data in "trains"
    dep_country = Column('dep_country', VARCHAR(3)) # store country short code of departure location
    dep_station_code = Column('dep_station_code', VARCHAR(10))# store departure station short code
    dep_time = Column('dep_time', DATETIME) # store time of departure 
    arr_country = Column('arr_country', VARCHAR(3)) # store country short code of arrival location
    arr_station_code = Column('arr_station_code', VARCHAR(10)) # store arrival station short code
    arr_time = Column('arr_time', DATETIME) # store time of arrival
    total_length = Column('total_length', Integer) # store length ot train set in total (incl. loco)
    max_speed = Column('max_speed', Integer) # store maximum permissable speed of train set
    
    # Defines relationship that allows to add instances of Journey_Section to instances of Train
    train = relationship('Train', backref=backref('journey_section'))
    
    
class Wagon(Base):
    '''Defines class object for table "wagon", see ER shema for details'''
    
    __tablename__ = 'wagon'
    
    uid = Column('id', Integer, autoincrement=True, primary_key=True) #id
    journey_id = Column('journey_id', Integer, ForeignKey('journey_section.id')) # id of corresponding
        # journey section
    wagon_type = Column('type', VARCHAR(25)) # store type of wagon
    loc = Column('loc', Integer) # store position (location) of wagon in train set
    sales_no = Column('sales_no', Integer) # store wagon number as presented to customer (reservations)
    length = Column('length', Integer) # store wagon length [cm] 
    playgr = Column('playgr', Boolean, default=False) # store whether wagon provides a playground
    disabled = Column('disabled', Boolean, default=False) # store whether wagon provides facilities 
        # for disabled
    catering = Column('catering', Boolean, default=False) # store whether wagon provides catering
    pet = Column('pet', Boolean, default=False) # store whether wagon provides facilties for pets
    luggage = Column('luggage', Boolean, default=False) # store whether wagon offers luggage racks
    vehicle_no = Column('vehicle_no', VARCHAR(13)) # store number of vehicle
    
    # Define relationship to be able to add instances of Wagon to instances of Journey_Section
    journey = relationship('Journey_Section', backref=backref('wagon'))
    
    def __repr__(self):
        '''Defines how print of instance of this class will look like.'''
        
        # dictionary of property names and properties of instance
        print_properties_dict = {'wagon_type': self.wagon_type, 
                                 'location': self.loc, 
                                 'sales_number': self.sales_no, 
                                 'length': self.length, 
                                 'playground': self.playgr, 
                                 'disabled': self.disabled, 
                                 'catering': self.catering, 
                                 'pet': self.pet, 
                                 'luggage': self.luggage, 
                                 'vehicle_number': self.vehicle_no}
        # emtpy f-string, will be filled and then returned
        string_repr = f''
        # loop over items in dictionary of properties and attach string representation to f-string
        for i, (key, item) in enumerate(print_properties_dict.items()):
            string_repr += f'{key}:\t{item}\n'
        # return f-string to be printed
        return string_repr + '\n'

    
class Locomotive(Base):
    '''Defines class object for table "locomotive", see ER shema for details'''
    
    __tablename__ = 'locomotive'
    
    uid = Column('id', Integer, autoincrement=True, primary_key=True) #id
    journey_id = Column('journey_id', Integer, ForeignKey('journey_section.id')) # id of corresponding 
    # journey section
    loco_type = Column('type', VARCHAR(25)) # store type of locomotive
    loc = Column('loc', Integer) # store position (location) of locomotive in train set
    power_type = Column('power_type', VARCHAR(25)) # store type of power of locomotive
    vehicle_no = Column('vehicle_no', Integer) # store number of vehicle
   
    # Define relationship to be able to add instances of Locomotive to instances of Journey_Section
    journey = relationship('Journey_Section', backref=backref('locomotive'))

def create_tables(db_str='mysql+mysqlconnector://root:admin123@my_sql_db/finrail'):
    '''Function returns engine to database specified in db_str and creates all tables form classes, 
    which inherited from "Base".
    
    Returns: sqlalchey database engine'''
    engine = create_engine(db_str) # Create engine
    Base.metadata.create_all(engine) # Create all tables
    return engine

def date_convert(date):
    '''Removes last letter from date string, then replaces every remaining letter with whitespace.
    
    Returns manipulated date string.'''
    date = re.sub('[A-z]$', '', date)
    return re.sub('[A-z]', ' ', date)

def trains_json_to_train_list(compositions_day):
    '''Function takes answer of API as specified on rata-digitraffic.fi for train compositions
    of whole days. 
    
    Returns list of Trains (sqlalchemy class object), ready to be send to finrail_db database.'''
    trains_of_day = []
    # Iterate through all trains in one day
    for train in compositions_day.json():
        t = Train() # Create instance of class Train (empty)
        # Fill t with values from json-object
        t.train_number = train['trainNumber']
        t.dep_date = train['departureDate']
        t.operator_code = train['operatorShortCode']
        t.train_cat = train['trainCategory']
        t.train_type = train['trainType']

        # Iterate through all journey sections in a train
        for section in train['journeySections']:
            j = Journey_Section() # Create instance of class Journey_Section (empty)
            # Fill j with values from json-object
            j.dep_country = section['beginTimeTableRow']['countryCode']
            j.dep_station_code = section['beginTimeTableRow']['stationShortCode']
            j.dep_time = date_convert(section['beginTimeTableRow']['scheduledTime'])
            j.arr_country = section['endTimeTableRow']['countryCode']
            j.arr_station_code = section['endTimeTableRow']['stationShortCode']
            j.arr_time = date_convert(section['endTimeTableRow']['scheduledTime'])
            j.total_length = section['totalLength']
            j.max_speed = section['maximumSpeed']

            # Iterate over all wagons in this journey section
            for wagon in section['wagons']:
                w = Wagon() # Create instance of Wagon (empty)
                # Iterate over all properties of wagon and fill instance w with values
                for i, (key, value) in enumerate(wagon.items()):
                    match key:
                        case 'wagonType':
                            w.wagon_type = value
                        case 'location':
                            w.loc = value
                        case 'salesNumber':
                            w.sales_no = value
                        case 'length':
                            w.length = value
                        case 'playground':
                            w.playgr = value
                        case 'disabled':
                            w.disabled = value
                        case 'catering':
                            w.catering = value
                        case 'pet':
                            w.pet = value
                        case 'luggage':
                            w.luggage = value
                        case 'vehicleNumber':
                            w.vehicle_no = value
                # Append w (instance of wagon) to wagon of this Journey_Section instance
                j.wagon.append(w)

            # Iterate over all locomotives in this journey section
            for loco in section['locomotives']:
                l = Locomotive() # Create instance of Locomotive
                # Iterate over all properties of loco and fill instance l with values
                for i, (key, value) in enumerate(loco.items()):
                    match key:
                        case 'locomotiveType':
                            l.loco_type = value
                        case 'location':
                            l.loc = value
                        case 'powerType':
                            l.power_type = value
                        case 'vehicleNumner':
                            l.vehicle_no = value
                # Append l (instance of locomotive) to locomotive of this Journey_Section instance
                j.locomotive.append(l)
            # Append j (instance of Train_Section) to journey_section of this train instance
            t.journey_section.append(j)
        trains_of_day.append(t)
    return trains_of_day

def dates_between(date_end, date_start=dt.date(2015, 12, 12)):
    '''Generator function that returns dates (data type: datetime.date).
    
    Parameters: 
    date_begin <datetime.date> First date in generator
    date_end <datetime.date> Date to end generator, caution this date is exclusive!
    
    Returns:
    Generator providing dates <datetime.date>
    '''
    while date_start < date_end:
        yield date_start
        date_start += dt.timedelta(days=1)
    
def add_compositions(s, date_end=dt.date.today(), verbose=0):
    '''Function will collect date of latest entries in finrail database. With this 
    information, it will fill up the database with the data from the rata.digitraffic.fi
    API for train compositions up to date_end (exclusive).
    
    Parameters: 
        s: sqlalchemy session instance
        date_end: datetime.date object (defaults to datetime.date.today())
        verbose: set to > 0 to obtain statis information while procession data
    '''
    # Query for latest date in database
    try:
        latest_date = s.query(func.max(Train.dep_date)).scalar()
    except:
        print('Table "train" in finrail database is not accessible, in "Query date". Set \
            date to default')
        latest_date = dt.date(2015, 12, 11) # default value for latest_date, if database not accessible
    if latest_date == None:
        latest_date = dt.date(2015, 12, 11) # default value for latest_date, if database is empty
    # Create generator for all dates missing in database up to today (excluding)
    gen_dates = dates_between(date_start=latest_date + dt.timedelta(days=1), 
                              date_end=date_end)
    # Iterate over all dates in generator and collect data from API and store in database
    for date in gen_dates:
        r = requests.get('https://rata.digitraffic.fi/api/v1/compositions/' + str(date))
        if r.status_code != 200: # Handle errors from requesting API
            print('API on rata.digitraffic.fi/api/v1/compositions/ is not accessible')
        else:
            try:
                # Extract data from answer of API and store in database
                s.add_all(trains_json_to_train_list(r))
                s.commit()
            except:
                # Handle errors that occured from accessing database
                print('finrail database is not accessible. In "Adding data"')
        if verbose > 0:
            # Print date of data just processed, if desired
            print('Added data of date: ' + str(date))