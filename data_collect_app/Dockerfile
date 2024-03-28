# Use alpine image of python to run python scripts with minimal load
FROM python:3.10.12-alpine

# Copy necessary files from project directory to /app/cron/ in image
COPY ./crontab ./finrail_db.py ./main.py ./sql_query.txt ./requirements.txt /app/cron/

# Install necessary python packages from requirements.txt file
RUN pip install -r /app/cron/requirements.txt
# Add entries in crontab file to crontab of image
RUN crontab /app/cron/crontab

# Define environment variables necessary to establish connection to a MySQL server
ENV DB_PSW default_psw
ENV DB_USER default_user
ENV DB_NAME default_name
ENV DB_SERVER default_server
# Define environment variable containing the file path of the app
ENV APP_PATH /app/cron/

# Define port
EXPOSE 5001

# start cron deamon to execute crontab and keep container running
CMD ["crond", "-f"]