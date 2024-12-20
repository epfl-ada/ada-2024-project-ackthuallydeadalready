import sys
import src.utils.preprocessing as pre
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler



def scatter_votes_year(df):
    '''
    scatter_votes_year(df)
    ## Function
    Plots the scatter of the number of votes per year between 2003 and 2013
    ## Variables
    df : dataset, contain a column 'YEA' with int from 2003 to 2013
    '''
    plt.scatter(np.linspace(2003,2013,11),df[['YEA']].value_counts(sort=False)) 
    return None

def boxplot_votes(df, src = 'SRC', vot = 'VOT'):
    '''
    boxplot_votes(df[, src, vot])
    ## Function
    Plots the number of votes per user or target in a box plot format.
    src is set to 'SRC' and vot to 'VOT' per default
    ## Variables
    df : dataset 
    src : string, name of the column containing the names ('SRC' or 'TGT')
    vot : string, name of column corresponding to the voter's approbation or disaproval
    '''
    plt.boxplot(df.groupby(src)[vot].count()) 
    return None

def comment_size(df): 
    '''
    comment_size(df)
    ## Function
    Plots the sizes of the comments in a bar plot. The comments are assumed in a column of name 'TXT'
    ## Variables
    df : dataset, contain a column 'TXT' with string
    '''
    sizeOcomment =df['TXT'].str.len() # Converts the series to str to be able to compute the length
    unique, counts = np.unique(sizeOcomment, return_counts=True) # Filters out comments that are simply a copy paste of another comment
    values = dict(zip(unique, counts))

    plt.bar(range(len(values)), list(values.values()), log=True)
    plt.show()
    return None

def plot_network(df, prt = False, savefig = False, path = './res/images/connexion.webp'):
    '''
    plot_network(df[, prt, savefig, path])
    ## Function
    Plots the connectivity network between votants and candidates, this takes a lot of time and the plot is heavy.
    prt and savefig are False by default
    Default path is './res/images/connexion.webp'
    ## Variables
    df : dataset, contain a column 'SRC' with string for the votants, 'TGT' for candidates, 'VOT' for the vote (+1 if positive, -1 negative)
    prt : bool, print the plot in the terminal /!\ heavy plot
    savefig : bool, save or not the figure
    path : string, path to save figure if savefig is True
    '''
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
        fig.write_image(path)
    return None



def visualize_cooperation(prt=False, save_fig=False, path='./res/images/cooperation.webp'):
    df = pre.complete_prepro_w_sa_topics()[0]
    df = df.dropna(subset='VOT')

    required_columns = ['SRC', 'TGT', 'VOT', 'YEA']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Dataframe must contain columns: {', '.join(required_columns)}")

    # Get unique years
    years = sorted(df['YEA'].unique())
    n_cols = 5
    n_rows = -(-len(years) // n_cols)  # Ceiling division for rows

    # Create subplots using Plotly
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[str(year) for year in years],  # Only show the year as the title
        vertical_spacing=0.1, horizontal_spacing=0.05
    )

    for idx, year in enumerate(years):
        # Filter data for the current year
        year_data = df[df['YEA'] == year]

        # Create a graph
        G = nx.Graph()
        for _, row in year_data.iterrows():
            src = row['SRC']
            tgt = row['TGT']
            vot = row['VOT']
            if G.has_edge(src, tgt):
                if G[src][tgt]['vote'] != vot:
                    G.remove_edge(src, tgt)
            G.add_edge(src, tgt, vote=vot)

        # Get node positions using a layout
        pos = nx.spring_layout(G, seed=42)
        edge_x = []
        edge_y = []
        edge_colors = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

            # Determine edge color based on vote
            vote = edge[2].get('vote', 0)
            if vote == 1:
                edge_colors.append('green')
            elif vote == -1:
                edge_colors.append('red')
            else:
                edge_colors.append('gray')

        # Create edge traces
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1),
            mode='lines',
            hoverinfo='none',
            marker=dict(color=edge_colors)  # Apply the color to edges
        )

        # Create node traces
        node_x = []
        node_y = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(
                size=4,  # Very small node size for better readability
                color='lightblue',
                line_width=1
            ),
            hoverinfo='none'  # Remove hover info to hide node names
        )

        # Add traces to the subplot
        row, col = divmod(idx, n_cols)
        fig.add_trace(edge_trace, row=row + 1, col=col + 1)
        fig.add_trace(node_trace, row=row + 1, col=col + 1)

    # Update layout for better presentation
    fig.update_layout(
        height=300 * n_rows, width=200 * n_cols,
        title_text="Cooperation Networks Over the Years",
        showlegend=False
    )

    if prt:
        fig.show()

    if save_fig:
        fig.write_image(path)

    return None




def visualize_cooperation_test(prt=False, save_fig=False, path='./res/images/cooperation.webp'):
    df = pre.complete_prepro_w_sa_topics()[0]

    df = df.dropna(subset='VOT')

    required_columns = ['SRC', 'TGT', 'VOT', 'YEA']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Dataframe must contain columns: {', '.join(required_columns)}")

    # Get unique years
    years = sorted(df['YEA'].unique())
    n_cols = 5
    n_rows = -(-len(years) // n_cols)  # Ceiling division for rows

    # Create subplots using Plotly
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[str(year) for year in years],  # Only show the year as the title
        vertical_spacing=0.1, horizontal_spacing=0.05
    )

    for idx, year in enumerate(years):
        # Filter data for the current year
        year_data = df[df['YEA'] == year]

                # Create graph
        G = nx.Graph()
        for _, row in df_test.iterrows():
            G.add_edge(row['SRC'], row['TGT'], vote=row['VOT'])

        # Get positions
        pos = nx.spring_layout(G, seed=42)  # You can adjust the seed or use another layout
        edge_x = []
        edge_y = []
        edge_colors = []

        # Create separate traces for each edge color
        edge_traces = []

        # Collect edge data
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            vote = edge[2].get('vote', 0)

            if vote == 1:
                edge_color = 'green'
            elif vote == -1:
                edge_color = 'red'
            else:
                edge_color = 'gray'

            # Create a separate edge trace for each edge
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1, color=edge_color),
                mode='lines',
                hoverinfo='none'
            )
            edge_traces.append(edge_trace)

            # Reset edge_x and edge_y for next iteration
            edge_x = []
            edge_y = []

        # Collect node data
        node_x = []
        node_y = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(
                size=10,  # Adjust the size of the nodes
                color='lightblue',
                line_width=2
            ),
            hoverinfo='none'
        )

        # Create figure and add traces
        fig = go.Figure(data=edge_traces + [node_trace])

    # Update layout for better presentation
    fig.update_layout(
        height=300 * n_rows, width=200 * n_cols,
        title_text="Cooperation Networks Over the Years",
        showlegend=False
    )

    if prt:
        fig.show()

    if save_fig:
        fig.write_image(path)

    return None


def plot_clusters(sn_clusters=5):
    """
    Plots clusters in a dataset using KMeans clustering, organized by sentiment, topic_x, and topic_y.
    
    Parameters:
        data (pd.DataFrame): The dataset containing the data to cluster.
        sentiment_col (str): The name of the sentiment column.
        topic_x_col (str): The name of the topic_x column.
        topic_y_col (str): The name of the topic_y column.
        n_clusters (int): The number of clusters for KMeans. Default is 5.
    """
    data=pre.complete_prepro_w_sa_topics()[0]

    # Encode categorical columns
    le_sentiment = LabelEncoder()
    le_topic_x = LabelEncoder()
    le_topic_y = LabelEncoder()
    
    data['sentiment_encoded'] = le_sentiment.fit_transform(data['sentiment'])
    data['topic_x_encoded'] = le_topic_x.fit_transform(data[topic_x])
    data['topic_y_encoded'] = le_topic_y.fit_transform(data[topic_y])
    
    # Select features for clustering
    features = ['sentiment_encoded', 'topic_x_encoded', 'topic_y_encoded']
    X = data[features]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA for 2D visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['cluster'] = kmeans.fit_predict(X_pca)
    
    # Plot clusters
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data['cluster'], cmap='viridis', s=50)
    plt.title('Clusters Based on Sentiment, Topic_x, and Topic_y', fontsize=16)
    plt.xlabel('PCA Component 1', fontsize=14)
    plt.ylabel('PCA Component 2', fontsize=14)
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
    
    # Add informative labels to clusters
    data['sentiment_label'] = le_sentiment.inverse_transform(data['sentiment_encoded'])
    data['topic_x_label'] = le_topic_x.inverse_transform(data['topic_x_encoded'])
    data['topic_y_label'] = le_topic_y.inverse_transform(data['topic_y_encoded'])
    
    if prt:
        fig.show()
    if save_fig:
        fig.write_image(path)

    return data[['sentiment_label', 'topic_x_label', 'topic_y_label', 'cluster']]


