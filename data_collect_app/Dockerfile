FROM python:3.10.12-alpine

COPY ./crontab ./finrail_db.py ./main.py ./requirements.txt /app/cron/

RUN pip install -r /app/cron/requirements.txt
RUN crontab /app/cron/crontab

CMD ["crond", "-f"]