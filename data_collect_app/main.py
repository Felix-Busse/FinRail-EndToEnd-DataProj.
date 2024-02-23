from crontab import CronTab

cron = CronTab(user=True)
job = cron.new('touch test.py')
job.minute.every(1)
cron.write()
job.run()