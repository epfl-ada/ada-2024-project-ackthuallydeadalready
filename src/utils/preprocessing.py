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
    '''
    parse_vote(vote_str)
    ## Function
    Parse each vote
    ## Variable
    vote_str : srting
    ## Return
    vote_dict : dict, containing parsed vote
    '''
    vote_dict = {}
    lines = vote_str.split('\n')
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            vote_dict[key.strip()] = value.strip()
    return vote_dict

def clean_txt_column(df):
    '''
    clean_txt_column(df)
    ## Function
    Remove ' and - characters
    ## Variable
    df : panda dataframe
    ## Return
    df : panda dataframe, without problematic characters
    '''
    df = df.str.replace(r"['-]+", "", regex=True)
    return df

def import_RFA():
    '''
    import_RFA()
    ## Function
    Import and parse the Request for Adminship original dataset
    ## Variables
    None
    ## Return
    df : pandas dataframe
    '''
    data = []
    with gzip.open('res/data/wiki-RfA.txt.gz', 'rt') as file:
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
    
def dates_prep(data_original, reorder=True):
    '''
    dates_prep(data_original[, reorder])
    ## Function
    ## Variables
    ## Return
    '''

    data_original['YEA'] = data_original['YEA'].astype(int)

    dates =[]
    for t in data_original['DAT']:
        dates.append(validate_datetime(t))

    dates = pd.DataFrame(dates)
    dates[dates[0]==False] = dates[dates[0]==False].replace(to_replace=False, value = float('NaN'))

    datetime_object_date = dates.apply(lambda x: x[0].date() if type(x[0])!=float else x[0], axis=1)
    datetime_object_month= dates.apply(lambda x: calendar.month_name[x[0].month] if type(x[0])!=float else x[0], axis=1)
    datetime_object_time = dates.apply(lambda x: x[0].time() if type(x[0])!=float else x[0], axis=1)

    # Converting types of DAT and adding TIM column
    data_original['DAT'] = datetime_object_date
    data_original['MON'] = datetime_object_month
    data_original['TIM'] = datetime_object_time
    # Reordering columns
    if reorder: 
        data_original = data_original.iloc[:,[0,1,2,3,4,5,7,8,6]]

    return data_original

def prep_unique_elections(data_original):
    '''
    prep_unique_elections(data_original)
    ## Function
    ## Variables
    ## Return
    '''
    uniq_elections = data_original.drop_duplicates(subset=['TGT','YEA','RES']).groupby('TGT').value_counts(dropna=False).reset_index()
    uniq_elections= uniq_elections.drop('count', axis=1)

    uniq_cand_freq_table= pd.DataFrame(uniq_elections['TGT'].value_counts())
    sing_runners_list = uniq_cand_freq_table[uniq_cand_freq_table['count']==1].index
    mult_runners_list = uniq_cand_freq_table[uniq_cand_freq_table['count']>1].index
    
    uniq_cand_freq_table = uniq_cand_freq_table.reset_index()

    return uniq_elections, uniq_cand_freq_table, sing_runners_list, mult_runners_list

def assign_elec_id(uniq_elections_data, complete_dataset): #i.e. (unique_elections_data, data)
    '''
    assign_elec_id(uniq_elections_data, complete_dataset)
    ## Function
    ## Variables
    ## Return
    '''
    uniq_elections_data['elec_id']= np.linspace(1,len(uniq_elections_data),len(uniq_elections_data))
    uniq_elections_data['elec_id']= uniq_elections_data['elec_id'].astype(int)
    
    complete_dataset['elec_id'] = np.empty(198275)
    complete_dataset['elec_id'] = complete_dataset['elec_id'].astype(int)

    for row in range(0,len(complete_dataset['TGT']),1):
        complete_dataset.iloc[row,9] = uniq_elections_data[(uniq_elections_data['TGT']==complete_dataset.iloc[row,1]) & 
                                        (uniq_elections_data['YEA']== complete_dataset.iloc[row,4]) &
                                        (uniq_elections_data['RES']== complete_dataset.iloc[row,3])]['elec_id']
    #print(data.iloc[row,[0,1,9]], row) #To Troubleshoot and find row which is giving error

    return uniq_elections_data, complete_dataset


def calc_win_loss(uniq_cand_freq_table, uniq_elections_data):
    '''
    calc_win_loss(uniq_cand_freq_table, uniq_elections_data)
    ## Function
    ## Variables
    ## Return
    '''
    win,loss =[],[]
    
    for row in range(0,len(uniq_cand_freq_table),1):
        if len(uniq_elections_data[uniq_elections_data['TGT']==uniq_cand_freq_table.iloc[row,0]][['TGT','RES']].value_counts().index) == 2:
            loss.append(uniq_elections_data[uniq_elections_data['TGT']==uniq_cand_freq_table.iloc[row,0]][['TGT','RES']].value_counts()[0])
            win.append(uniq_elections_data[uniq_elections_data['TGT']==uniq_cand_freq_table.iloc[row,0]][['TGT','RES']].value_counts()[1])
        elif len(uniq_elections_data[uniq_elections_data['TGT']==uniq_cand_freq_table.iloc[row,0]][['TGT','RES']].value_counts().index) ==1 and\
        uniq_elections_data[uniq_elections_data['TGT']==uniq_cand_freq_table.iloc[row,0]][['TGT','RES']].value_counts().index[0][1] == -1:
            loss.append(uniq_elections_data[uniq_elections_data['TGT']==uniq_cand_freq_table.iloc[row,0]][['TGT','RES']].value_counts()[0])
            win.append(0)
        elif len(uniq_elections_data[uniq_elections_data['TGT']==uniq_cand_freq_table.iloc[row,0]][['TGT','RES']].value_counts().index) ==1 and\
        uniq_elections_data[uniq_elections_data['TGT']==uniq_cand_freq_table.iloc[row,0]][['TGT','RES']].value_counts().index[0][1] == 1:
            loss.append(0)
            win.append(uniq_elections_data[uniq_elections_data['TGT']==uniq_cand_freq_table.iloc[row,0]][['TGT','RES']].value_counts()[0])
        else:
            print('Something unexpected happen')

    uniq_cand_freq_table['win'] = win
    uniq_cand_freq_table['loss'] = loss
    return uniq_cand_freq_table

def flag_elec_id(uniq_elections_data, complete_dataset):
    '''
    flag_elec_id(uniq_elections_data, complete_dataset)
    ## Function
    ## Variables
    ## Return
    '''
    flagged = []

    for elec_id in uniq_elections_data['elec_id']:
        #check if same month, next or previous month as mode
        for idx in complete_dataset[complete_dataset['elec_id']==elec_id].index:
            if type(complete_dataset.iloc[idx,5]) == float:
                #print('skipped NaN')
                continue

            elif (datetime.strptime(complete_dataset[complete_dataset['elec_id']==elec_id]['MON'].mode()[0], '%B').month) == 1:
                if (datetime.strptime(complete_dataset[complete_dataset['elec_id']==elec_id]['MON'].mode()[0], '%B').month == complete_dataset.iloc[idx,5].month) or \
                    (12 == complete_dataset.iloc[idx,5].month) or \
                    (2 == complete_dataset.iloc[idx,5].month):
                    pass
                    #print('No Problem - Jan', idx)

            elif (datetime.strptime(complete_dataset[complete_dataset['elec_id']==elec_id]['MON'].mode()[0], '%B').month) == 12:
                if (datetime.strptime(complete_dataset[complete_dataset['elec_id']==elec_id]['MON'].mode()[0], '%B').month == complete_dataset.iloc[idx,5].month) or \
                    (11 == complete_dataset.iloc[idx,5].month) or \
                    (1 == complete_dataset.iloc[idx,5].month):
                    pass
                    #print('No Problem- Dec', idx)

            elif (datetime.strptime(complete_dataset[complete_dataset['elec_id']==elec_id]['MON'].mode()[0], '%B').month) != 1 and 2:
                if (datetime.strptime(complete_dataset[complete_dataset['elec_id']==elec_id]['MON'].mode()[0], '%B').month == complete_dataset.iloc[idx,5].month) or \
                    ((datetime.strptime(complete_dataset[complete_dataset['elec_id']==elec_id]['MON'].mode()[0], '%B').month+1) == complete_dataset.iloc[idx,5].month) or \
                    ((datetime.strptime(complete_dataset[complete_dataset['elec_id']==elec_id]['MON'].mode()[0], '%B').month-1) == complete_dataset.iloc[idx,5].month):
                    pass
                    #print('No Problem', idx)
                else:
                    #print('row', idx)
                    flagged.append([elec_id, idx])
    
    flagged = pd.DataFrame(flagged, columns=('elec_id','index'))

    return flagged

def get_flagged_boundaries(flagged):
    '''
    get_flagged_boundaries(flagged)
    ## Function
    ## Variables
    ## Return
    '''
    flagged_with_boundaries = (flagged.groupby('elec_id').min()).merge(flagged.groupby('elec_id').max(), on='elec_id', suffixes=('_lower','_upper'))

    return flagged_with_boundaries

def separate_flagged_elec_id(complete_dataset, flagged):
    '''
    separate_flagged_elec_id(complete_dataset, flagged)
    ## Function
    ## Variables
    ## Return
    '''
    x = complete_dataset.copy()
    flagged_with_bounds = get_flagged_boundaries(flagged=flagged)

    for elec_id in flagged_with_bounds.index:
        try:    
            x[x['elec_id']==elec_id].loc[flagged_with_bounds.loc[elec_id]['index_lower']-1][9]
            #print('flagged election is after, shift elec_id from lower bound')
            
            # Case 1: flagged election is after, shift elec_id from lower bound
            for idx in x.index:
                if x.iloc[idx,9]> elec_id:
                    x.iloc[idx,9] = x.iloc[idx,9]+1

            for idx in range(flagged_with_bounds.index_lower.loc[elec_id], flagged_with_bounds.index_upper.loc[elec_id]+1,1):
                x.iloc[idx,9] = x.iloc[idx,9]+1


        except KeyError:
            #print('flagged election is before, shift elec_id above upper bound')

            # Case 2: flagged election is before, shift elec_id above upper bound
            for idx in x.index:
                if x.iloc[idx,9]>= elec_id:
                    x.iloc[idx,9] = x.iloc[idx,9]+1

            for idx in range(flagged_with_bounds.index_lower.loc[elec_id], flagged_with_bounds.index_upper.loc[elec_id]+1,1):
                x.iloc[idx,9] = x.iloc[idx,9]-1
        
    file_path = 'data/processed_elec_data.csv'
    x.to_csv(file_path, index=False)

    return x

def get_votes(complete_dataset):
    '''

    '''
    votes_by_elec = pd.DataFrame(complete_dataset.groupby('elec_id')['RES'].agg('count'))
    votes_by_elec = votes_by_elec.rename(columns={'RES':'TOT'})
    votes_by_elec = votes_by_elec.reset_index()

    pos = complete_dataset[complete_dataset['VOT']==1].groupby('elec_id')['VOT'].agg('count').reset_index()
    pos = pos.rename(columns={'VOT':'POS'})
    #pos

    abs = complete_dataset[complete_dataset['VOT']==0].groupby('elec_id')['VOT'].agg('count').reset_index()
    abs = abs.rename(columns={'VOT':'ABS'})
    #abs

    neg = complete_dataset[complete_dataset['VOT']==-1].groupby('elec_id')['VOT'].agg('count').reset_index()
    neg = neg.rename(columns={'VOT':'NEG'})
    #abs

    votes_by_elec = votes_by_elec.merge(pos, on='elec_id', how='left')
    votes_by_elec = votes_by_elec.merge(abs, on='elec_id', how='left')
    votes_by_elec = votes_by_elec.merge(neg, on='elec_id', how='left')

    votes_by_elec= votes_by_elec.fillna(0)


    votes_by_elec['POS'] = votes_by_elec['POS'].astype(int)
    votes_by_elec['ABS'] = votes_by_elec['ABS'].astype(int)
    votes_by_elec['NEG'] = votes_by_elec['NEG'].astype(int)

    complete_dataset = complete_dataset.merge(votes_by_elec, how='left',on='elec_id')

    return votes_by_elec, complete_dataset

def preprossess_eda(data, prepdates = True,impor = True):
    '''
    >12min
    '''
    if prepdates:
        df = dates_prep(data)
    else :
        df = data.copy()
    unique_elections, unique_candidate_freq_table, single_runners_list, multiple_runners_list  = prep_unique_elections(df)
    unique_elections, df = assign_elec_id(unique_elections,df)
    unique_candidate_freq_table = calc_win_loss(unique_candidate_freq_table,unique_elections)
    flagged_elec_id = flag_elec_id(unique_elections, df)
    if impor :
        file_path = 'res/data/processed_elec_data.csv'
        df_processed = pd.read_csv(file_path)
    else :
        df_processed = separate_flagged_elec_id(df,flagged_elec_id)
    return df_processed, df, unique_elections, unique_candidate_freq_table, single_runners_list, multiple_runners_list, flagged_elec_id

def merge_sa_eda(data_eda,data_sa):
    merged = data_eda.merge(data_sa[['sentiment', 'vader_neg', 'vader_neu', 'vader_pos', 'vader_compound', 'topic_x', 'topic_y']], how='left')
    return merged

def single_multi_stat(unique_elections,single_runners_list,multiple_runners_list):
    '''
    '''
    sing_mult_stats = pd.DataFrame({'SING' : unique_elections[unique_elections['TGT'].isin(single_runners_list)]['RES'].value_counts(normalize=False),
                'SING_PERC' : unique_elections[unique_elections['TGT'].isin(single_runners_list)]['RES'].value_counts(normalize=True),
                'MULT' : unique_elections[unique_elections['TGT'].isin(multiple_runners_list)]['RES'].value_counts(normalize=False),
                'MULT_PERC' : unique_elections[unique_elections['TGT'].isin(multiple_runners_list)]['RES'].value_counts(normalize=True)
    })

    sing_mult_stats_total = pd.DataFrame(sing_mult_stats.sum(axis=0)).rename(columns={0:'Total'}).transpose().astype(int)

    sing_mult_stats = pd.concat([sing_mult_stats,sing_mult_stats_total], axis=0)
    return sing_mult_stats

def passrate_byYear(df1):
    passrate_by_year = pd.DataFrame({'WIN': df1.drop_duplicates('elec_id').reset_index().drop('index', axis=1).groupby('YEA')['RES'].value_counts(normalize=False).unstack().fillna(0).iloc[:,1].astype(int),
                                    'LOSS': df1.drop_duplicates('elec_id').reset_index().drop('index', axis=1).groupby('YEA')['RES'].value_counts(normalize=False).unstack().fillna(0).iloc[:,0].astype(int),
                                    'WIN_PERC': df1.drop_duplicates('elec_id').reset_index().drop('index', axis=1).groupby('YEA')['RES'].value_counts(normalize=True).unstack().fillna(0).iloc[:,1],
                                    'LOSS_PERC': df1.drop_duplicates('elec_id').reset_index().drop('index', axis=1).groupby('YEA')['RES'].value_counts(normalize=True).unstack().fillna(0).iloc[:,0],
    })

    passrate_by_year.reset_index()

    return passrate_by_year

def complete_prepro_w_sa_topics(df_path = "res/data/rfa_bert_vader_topic.csv", 
    qs_path = 'res/data/all_questions_and_answers_w_topic.csv', thr_tp=True):
    '''
    prep_unique_elections(data_original)
    ## Function
    ## Variables
    ## Return

    takes csv vader and hug sa, preprocess it, return preprossed dataframe with additional features and questions/answers topics
    '''
    qs = pd.read_csv(qs_path)
    qs['Question'] = qs['Question'].astype(str)
    qs['Answer'] = qs['Answer'].astype(str)
    qs['User'] = qs['User'].astype(str)
    if thr_tp:
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
    df['prop_pos_vot']= df['pos_votes']/(df['pos_votes']+df['neu_votes']+df['neg_votes'])

    qs.rename(columns={'User': 'TGT'}, inplace=True)
    topic_sets = qs.groupby(['TGT', 'Attempt'])['topic'].apply(lambda x: set(x)).reset_index()
    df= df.merge(topic_sets, on=['TGT', 'Attempt'], how='left')


    return df, qs

