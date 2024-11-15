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
    plt.scatter(np.linspace(2003,2013,11),df[['YEA']].value_counts(sort=False)) # Plots the number of votes per year between the 
                                                                                # years 2003 and 2013
    return None

def boxplot_byUser(df):
    plt.boxplot(df.groupby('SRC')['VOT'].count()) # Plots the number of votes per user in a box plot format, the 1601 outlier is 
                                                  # the aggregate of the people whose username was not recorded during their vote
    return None

def boxplot_byTarget(df):
    plt.boxplot(df.groupby('TGT')['VOT'].count()) # Plots the number of votes each candidate has received in a box plot
    return None


def comment_size(df): # Plots the sizes of the comments in a bar plot
    sizeOcomment =df['TXT'].str.len() # Converts the series to str to be able to compute the length
    unique, counts = np.unique(sizeOcomment, return_counts=True) # Filters out comments that are simply a copy paste of another comment
    values = dict(zip(unique, counts))

    plt.bar(range(len(values)), list(values.values()), log=True)
    plt.show()
    return None
