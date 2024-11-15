import sys
import src.Data.preprocessing as pre
import matplotlib.pyplot as plt
import numpy as np

def scraping_res():
    all_names, answer_one = pre.scrape()

    for key, value in all_names["(aeropagitica)"].items():
        print(key)
        print(value)
    
    return None

def plot_byYear(df):
    plt.scatter(np.linspace(2003,2013,11),df[['YEA']].value_counts(sort=False))
    return None

def boxplot_byUser(df):
    plt.boxplot(df.groupby('SRC')['VOT'].count())
    return None

def boxplot_byTarget(df):
    plt.boxplot(df.groupby('TGT')['VOT'].count())
    return None


def comment_size(df):
    sizeOcomment =df['TXT'].str.len()
    unique, counts = np.unique(sizeOcomment, return_counts=True)
    values = dict(zip(unique, counts))

    plt.bar(range(len(values)), list(values.values()), log=True)
    plt.show()
    return None
