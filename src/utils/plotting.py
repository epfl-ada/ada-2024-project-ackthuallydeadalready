import sys
import src.utils.preprocessing as pre
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import plotly.graph_objects as go

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

def plot_network(df, prt = False, savefig = False):
    G = nx.DiGraph()

    for _, row in df.iterrows():
        G.add_edge(row['SRC'], row['TGT'], weight=row['VOT'])

    pos = nx.spring_layout(G)  # or nx.circular_layout(G)

    # Extract node positions
    node_x = [pos[node][0] for node in G.nodes]
    node_y = [pos[node][1] for node in G.nodes]

    # Add nodes to the plot
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(size=10, color='gray'),
    )

    # Add edges to the plot
    edge_x = []
    edge_y = []
    edge_colors = []

    for edge in G.edges(data=True):
        src, tgt, attrs = edge
        edge_x += [pos[src][0], pos[tgt][0], None]
        edge_y += [pos[src][1], pos[tgt][1], None]
        edge_colors.append('green' if attrs['weight'] == 1 else 'red')

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color=edge_colors),
        hoverinfo='none',
        mode='lines'
    )

    # Create the final figure
    fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='User Connections',
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=0, l=0, r=0, t=0),
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=False, zeroline=False)
                ))
    if prt:
        fig.show()
    if savefig :
        fig.write_image('./Plots/connexion_graph.webp')
    return None
