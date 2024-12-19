import sys
import numpy as np
import pandas as pd
import gzip
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from datetime import datetime
import calendar 

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
    
#Two different Formats identified in the DAT column with typos
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

def dates_prep(data, reorder=True):
    data['YEA'] = data['YEA'].astype(int)
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
    if reorder :
        data = data.iloc[:,[0,1,2,3,4,5,7,8,6]]

    return data

def prep_unique_elections(data_original):
    unique_elections = data_original.drop_duplicates(subset=['TGT','YEA','RES']).groupby('TGT').value_counts(dropna=False).reset_index()
    unique_elections= unique_elections.drop('count', axis=1)

    unique_candidate_freq_table= pd.DataFrame(unique_elections['TGT'].value_counts())
    single_runners_list = unique_candidate_freq_table[unique_candidate_freq_table['count']==1].index
    multiple_runners_list = unique_candidate_freq_table[unique_candidate_freq_table['count']>1].index
    
    unique_candidate_freq_table = unique_candidate_freq_table.reset_index()

    return unique_elections

def prep_elec_id(unique_elections_data):
    unique_elections_data['elec_id']= np.linspace(1,len(unique_elections_data),len(unique_elections_data))
    unique_elections_data['elec_id']= unique_elections_data['elec_id'].astype(int)
    
    data['elec_id']= np.empty(198275)
    data['elec_id']= data['elec_id'].astype(int)

    for row in range(0,len(data['TGT']),1):
        data.iloc[row,9] = unique_elections[(unique_elections['TGT']==data.iloc[row,1]) & 
                                        (unique_elections['YEA']== data.iloc[row,4]) &
                                        (unique_elections['RES']== data.iloc[row,3])]['elec_id']
    #print(data.iloc[row,[0,1,9]], row) #To Troubleshoot and find row which is giving error

    win,loss =[],[]
    
    for row in range(0,len(unique_candidate_freq_table),1):
        if len(unique_elections[unique_elections['TGT']==unique_candidate_freq_table.iloc[row,0]][['TGT','RES']].value_counts().index) == 2:
            loss.append(unique_elections[unique_elections['TGT']==unique_candidate_freq_table.iloc[row,0]][['TGT','RES']].value_counts()[0])
            win.append(unique_elections[unique_elections['TGT']==unique_candidate_freq_table.iloc[row,0]][['TGT','RES']].value_counts()[1])
        elif len(unique_elections[unique_elections['TGT']==unique_candidate_freq_table.iloc[row,0]][['TGT','RES']].value_counts().index) ==1 and\
        unique_elections[unique_elections['TGT']==unique_candidate_freq_table.iloc[row,0]][['TGT','RES']].value_counts().index[0][1] == -1:
            loss.append(unique_elections[unique_elections['TGT']==unique_candidate_freq_table.iloc[row,0]][['TGT','RES']].value_counts()[0])
            win.append(0)
        elif len(unique_elections[unique_elections['TGT']==unique_candidate_freq_table.iloc[row,0]][['TGT','RES']].value_counts().index) ==1 and\
        unique_elections[unique_elections['TGT']==unique_candidate_freq_table.iloc[row,0]][['TGT','RES']].value_counts().index[0][1] == 1:
            loss.append(0)
            win.append(unique_elections[unique_elections['TGT']==unique_candidate_freq_table.iloc[row,0]][['TGT','RES']].value_counts()[0])
        else:
            print('Something unexpected happen')

    unique_candidate_freq_table['win'] = win
    unique_candidate_freq_table['loss'] = loss
    return unique_candidate_freq_table


def complete_prepro_w_sa_topics(df_path = "data/rfa_bert_vader_topic.csv", qs_path = 'data/all_questions_and_answers_w_topic.csv'):
    qs = pd.read_csv(qs_path)
    qs['Question'] = qs['Question'].astype(str)
    qs['Answer'] = qs['Answer'].astype(str)
    threshold = 10
    topic_counts = qs['topic'].value_counts()
    valid_topics = topic_counts[topic_counts >= threshold].index
    qs = qs[qs['topic'].isin(valid_topics)]
    df=pd.read_csv(df_path, index_col=0)
    df['TXT'] = df['TXT'].astype(str)
    df['DAT'] = df['DAT'].astype(str)
    df['SRC'] = df['SRC'].fillna('anonymous voters')
    df=dates_prep(df, reorder=False)
    unique_attempts = pd.DataFrame(df.drop_duplicates(subset=['TGT','YEA','RES'])).sort_values(by=['TGT', 'YEA'])
    unique_attempts['Attempt'] = unique_attempts.groupby('TGT').cumcount() + 1
    df = df.merge(unique_attempts[['TGT', 'YEA','RES', 'Attempt']], on=['TGT', 'YEA','RES'], how='left')
    df['total_vote_count'] = df.groupby(['TGT', 'Attempt'])['VOT'].transform('count')
    df['pos_votes'] = df.groupby(['TGT', 'Attempt'])['VOT'].transform(lambda x: (x == 1).sum())
    df['neu_votes'] = df.groupby(['TGT', 'Attempt'])['VOT'].transform(lambda x: (x == 0).sum())
    df['neg_votes'] = df.groupby(['TGT', 'Attempt'])['VOT'].transform(lambda x: (x == -1).sum())
    return df, qs

