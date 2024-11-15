import sys
import src.Data.preprocessing as pre
import matplotlib.pyplot as plt

def scrapping_res():
    all_names, answer_one = pre.scrape()

    for key, value in all_names["(aeropagitica)"].items():
        print(key)
        print(value)

def plot_byYear(df):
    plt.hist(df[['YEA']].loc[['2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013']])
    return None

def boxplot_byUser(df):
    plt.boxplot(df.groupby('SRC')['VOT'].count())
    return None

def hist_votes(df):
    plt.hist(df[['VOT']].value_counts())
    return None


# def plot_byYear(df):
#     data = pd.DataFrame({
#         '2003' : df[df['YEA']==2003]['SRC'],
#         '2004' : df[df['YEA']==2004]['SRC'],
#         '2005' : df[df['YEA']==2005]['SRC'],
#         '2006' : df[df['YEA']==2006]['SRC'],
#         '2007' : df[df['YEA']==2007]['SRC'],
#         '2008' : df[df['YEA']==2008]['SRC'],
#         '2009' : df[df['YEA']==2009]['SRC'],
#         '2010' : df[df['YEA']==2010]['SRC'],
#         '2011' : df[df['YEA']==2011]['SRC'],
#         '2012' : df[df['YEA']==2012]['SRC'],
#         '2013' : df[df['YEA']==2013]['SRC']
#     })
#     data.value_counts().hist()
#     return None