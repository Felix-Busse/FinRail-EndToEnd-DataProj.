# Use image of python to run python scripts
#FROM python:3.10.14-bookworm
FROM tensorflow/tensorflow:2.14.0

# copy necessary files from development folder to image
COPY main.py finrail_rnn_model.py timeseries_query.txt rnn_commuter.keras rnn_long_distance.keras requirements.txt /app/finrail_api/

# Update pip (in order to find desired version of tensorflow)
RUN pip install --upgrade pip

# Install dependencies of python code via pip
RUN pip install -r ./app/finrail_api/requirements.txt

# Define environment variables necessary to establish connection to a MySQL server
ENV DB_PSW default_psw
ENV DB_USER default_user
ENV DB_NAME default_name
ENV DB_SERVER default_server
# Define environment variable for port number of image and uvicorn server in container
# And set it to value of ARG variable (only valid during build)
ARG ARG_PORT_NO
ENV PORT_NO=$ARG_PORT_NO

# Define port number
EXPOSE $PORT_NO

# Start API on uvicorn server
CMD uvicorn main:app --host 0.0.0.0 --port $PORT_NO --app-dir app/finrail_api