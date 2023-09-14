import pandas as pd

day_cat = [
    'Monday',
    'Tuesday',
    'Wednesday',
    'Thursday',
    'Friday',
    'Saturday',
    'Sunday'
]

day_of_week_type = pd.CategoricalDtype(categories=day_cat, ordered=True)
