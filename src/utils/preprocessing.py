import sys
import numpy as np
import pandas as pd
import gzip
import matplotlib.pyplot as plt
import seaborn as sns

# Parsing and importing datas
def parse_vote(vote_str):
    vote_dict = {}
    lines = vote_str.split('\n')
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            vote_dict[key.strip()] = value.strip()
    return vote_dict

def clean_txt_column(df):
    # Remove ''' - and -- characters
    df = df.str.replace(r"['-]+", "", regex=True)
    return df

def import_RFA():
    data = []
    with gzip.open('data/wiki-RfA.txt.gz', 'rt') as file:
        content = file.read()

        # Split votes using blank lines
        votes = content.strip().split('\n\n')

        # Process each vote
        for vote in votes:
            vote_data = parse_vote(vote)
            data.append(vote_data)

    df = pd.DataFrame(data)

    df['VOT'] = pd.to_numeric(df['VOT'],errors='coerce')
    df['RES'] = pd.to_numeric(df['RES'],errors='coerce')
    #df['YEA'] = pd.to_numeric(df['YEA'],errors='coerce')
    df['TXT']=clean_txt_column(df['TXT'])

    return df

#Error handling code from Stacksoverflow: "validate for multiple date / datetime formats" by 'c8999c 3f964f64'
def validate_datetime(string, whitelist=('%H:%M, %d %B %Y', '%H:%M, %d %b %Y')):
    for fmt in whitelist:
        try:
            dt = datetime.strptime(string, fmt)
        except ValueError:
            pass
        else: # if a defined format is found, datetime object will be returned
            return dt
    else: # all formats done, none did work...
        return False # could also raise an exception here

def dates_prep(data):
    data['YEA'] = data['YEA'].astype(int)
    dates =[]
    for t in data['DAT']:
        dates.append(validate_datetime(t))

    dates = pd.DataFrame(dates)
    dates[dates[0]==False] = dates[dates[0]==False].replace(to_replace=False, value = float('NaN'))

    datetime_object_date = dates.apply(lambda x: x[0].date() if type(x[0])!=float else x[0], axis=1)
    datetime_object_month= dates.apply(lambda x: calendar.month_name[x[0].month] if type(x[0])!=float else x[0], axis=1)
    datetime_object_time = dates.apply(lambda x: x[0].time() if type(x[0])!=float else x[0], axis=1)

    # Converting types of DAT and adding TIM column
    data['DAT'] = datetime_object_date
    data['MON'] = datetime_object_month
    data['TIM'] = datetime_object_time
    # Reordering columns
    data = data.iloc[:,[0,1,2,3,4,5,7,8,6]]

    return data
