# Use image of python to run python scripts
#FROM python:3.10.14-bookworm
FROM tensorflow/tensorflow:2.14.0

# Define environment variable for app directory
ENV APP_DIR=/app/finrail_api/

# copy necessary files from development folder to image
COPY main.py finrail_rnn_model.py timeseries_query.txt rnn_commuter.keras \
    rnn_long_distance.keras requirements.txt ${APP_DIR}

# Install dependencies of python code via pip
RUN pip install -r .${APP_DIR}requirements.txt

# Define environment variables necessary to establish connection to 
# a MySQL server
ENV DB_PSW=default_psw
ENV DB_USER=default_user
ENV DB_NAME=default_name
ENV DB_SERVER=default_server
# Define environment variable for port number of image and uvicorn server 
# in container
# And set it to value of ARG variable (only valid during build)
ARG ARG_PORT_NO
ENV PORT_NO=$ARG_PORT_NO

# Define port number
EXPOSE $PORT_NO

# Start API on uvicorn server
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT_NO} --app-dir ${APP_DIR}