import sys
import src.utils.preprocessing as pre
import matplotlib.pyplot as plt
import numpy as np

def plot_byYear(df):
    '''
    plot_byYear(df)
    ## Function
    Plots the scatter of the number of votes per year between 2003 and 2013
    ## Variables
    df : dataset with a column 'YEA' containing int from 2003 to 2013
    '''
    plt.scatter(np.linspace(2003,2013,11),df[['YEA']].value_counts(sort=False)) 
    return None

def boxplot_byUser(df, src = 'SRC', vot = 'VOT'):
    '''
    plot_byUser(df[, src, vot])
    ## Function
    Plots the number of votes per user in a box plot format.
    src is set to 'SRC' and vot to 'VOT' per default
    ## Variables
    df : dataset 
    src : string of the name of the column containing string of the user 
    vot : string of a binary column corresponding to the voter's approbation or disaproval
    '''
    plt.boxplot(df.groupby(src)[vot].count()) 
    return None

def boxplot_byTarget(df, tgt = 'TGT', vot = 'VOT'):
    '''
    plot_byTarget(df[, tgt, vot])
    ## Function
    Plots the number of votes each candidate has received in a box plot.
    tgt is set to 'TGT' and vot to 'VOT' per default
    ## Variables
    df : dataset 
    tgt : string of the name of the column containing string of the target
    vot : string of a binary column corresponding to the voter's approbation or disaproval
    '''
    plt.boxplot(df.groupby(tgt)[vot].count()) 
    return None

def comment_size(df): 
    '''
    plot_byTarget(df)
    ## Function
    Plots the sizes of the comments in a bar plot. The comments are assumed in a column of name 'TXT'
    ## Variables
    df : dataset
    '''
    sizeOcomment =df['TXT'].str.len() # Converts the series to str to be able to compute the length
    unique, counts = np.unique(sizeOcomment, return_counts=True) # Filters out comments that are simply a copy paste of another comment
    values = dict(zip(unique, counts))

    plt.bar(range(len(values)), list(values.values()), log=True)
    plt.show()
    return None
