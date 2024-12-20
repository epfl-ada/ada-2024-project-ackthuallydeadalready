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
import plotly.io as pio
import src.utils.plotting as plot
from matplotlib.lines import Line2D
from sklearn.preprocessing import MultiLabelBinarizer



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



def plot_sentiment_byPass(data, vader=False, prt = False, savefig = False):
    if vader :
        data['sentiment_vader'] = data.apply(
            lambda row: 'POSITIVE' if row['vader_pos'] > row['vader_neg'] else 'NEGATIVE', axis=1
        )

        sentiment_stats_vader = data.groupby(['sentiment_vader', 'RES']).size().unstack(fill_value=0)

        # Calculate total and success rate by sentiment
        sentiment_stats_vader['Total'] = sentiment_stats_vader.sum(axis=1)
        sentiment_stats_vader['Pass Rate'] = sentiment_stats_vader[1] / sentiment_stats_vader['Total']

        # Format for plotting
        df_g = sentiment_stats_vader[[1, -1, 'Total']].reset_index().rename(columns={1: 'pos', -1: 'neg'})
        df_g2 = sentiment_stats_vader[['Pass Rate']].reset_index()

        fig = go.Figure(
            data=[
                # Bar for Losses
                go.Bar(
                    x=df_g['sentiment_vader'],
                    y=df_g['neg'],
                    name="Losses",
                    text=df_g['neg'],
                    textposition='inside',
                    marker=dict(color="salmon")
                ),
                # Bar for Wins
                go.Bar(
                    x=df_g['sentiment_vader'],
                    y=df_g['pos'],
                    name="Wins",
                    text=df_g['pos'],
                    textposition='inside',
                    marker=dict(color="teal")
                ),
                # Bar for Totals
                go.Bar(
                    x=df_g['sentiment_vader'],
                    y=df_g['Total'],
                    name="Total",
                    text=df_g['Total'],
                    textposition='outside',
                    marker=dict(color="blue")
                ),
            ],
            layout=dict(bargap=0.2,barcornerradius=15)
        )

        # Overlay line graph for Pass Rate
        fig.add_trace(
            go.Scatter(
                x=df_g2['sentiment_vader'],
                y=df_g2['Pass Rate'] * 100,  # Convert to percentage
                mode="lines+markers+text",
                name="Pass Rate",
                text=[f"{pr * 100:.1f}%" for pr in df_g2["Pass Rate"]],
                textposition="top center",
                line=dict(color="red", width=2),
                marker=dict(color="red", size=8),
                yaxis="y2"
            )
        )

        # Update layout
        fig.update_layout(
            title=dict(
                text="Success Rate vs Voter Sentiment",
                x=0.5,
                xanchor="center"
            ),
            xaxis=dict(
                title="Voter Sentiment"
            ),
            yaxis=dict(
                title="Counts",
                side="left"
            ),
            yaxis2=dict(
                title="Pass Rate (%)",
                overlaying="y",
                side="right",
                range=[0, 100],
                tickformat=".0f"
            ),
            barmode="group",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        if prt:
            fig.show()
        if savefig :
            pio.write_html(fig, file="res/Plots/pass_rates_positive_negative_vader.html", auto_open=False)
    else :
        sentiment_stats = data.groupby(['sentiment', 'RES']).size().unstack(fill_value=0)

        # Calculate total and success rate by sentiment
        sentiment_stats['Total'] = sentiment_stats.sum(axis=1)
        sentiment_stats['Pass Rate'] = sentiment_stats[1] / sentiment_stats['Total']

        # Format for plotting
        df_g = sentiment_stats[[1, -1, 'Total']].reset_index().rename(columns={1: 'pos', -1: 'neg'})
        df_g2 = sentiment_stats[['Pass Rate']].reset_index()

        fig = go.Figure(
            data=[
                # Bar for Losses
                go.Bar(
                    x=df_g['sentiment'],
                    y=df_g['neg'],
                    name="Losses",
                    text=df_g['neg'],
                    textposition='inside',
                    marker=dict(color="salmon")
                ),
                # Bar for Wins
                go.Bar(
                    x=df_g['sentiment'],
                    y=df_g['pos'],
                    name="Wins",
                    text=df_g['pos'],
                    textposition='inside',
                    marker=dict(color="teal")
                ),
                # Bar for Totals
                go.Bar(
                    x=df_g['sentiment'],
                    y=df_g['Total'],
                    name="Total",
                    text=df_g['Total'],
                    textposition='outside',
                    marker=dict(color="blue")
                ),
            ],
            layout=dict(bargap=0.2,barcornerradius=15)
        )

        # Overlay line graph for Pass Rate
        fig.add_trace(
            go.Scatter(
                x=df_g2['sentiment'],
                y=df_g2['Pass Rate'] * 100,  # Convert to percentage
                mode="lines+markers+text",
                name="Pass Rate",
                text=[f"{pr * 100:.1f}%" for pr in df_g2["Pass Rate"]],
                textposition="top center",
                line=dict(color="red", width=2),
                marker=dict(color="red", size=8),
                yaxis="y2"
            )
        )

        # Update layout
        fig.update_layout(
            title=dict(
                text="Success Rate vs Voter Sentiment",
                x=0.5,
                xanchor="center"
            ),
            xaxis=dict(
                title="Voter Sentiment"
            ),
            yaxis=dict(
                title="Counts",
                side="left"
            ),
            yaxis2=dict(
                title="Pass Rate (%)",
                overlaying="y",
                side="right",
                range=[0, 100],
                tickformat=".0f"
            ),
            barmode="group",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        if prt:
            fig.show()
        if savefig :
            pio.write_html(fig, file="res/Plots/pass_rates_positive_negative.html", auto_open=False)
    return None



def plot_sentiments_byYear(data, vader=False, prt = False, savefig = False):
    if vader:
        data['sentiment_vader'] = data.apply(
            lambda row: 'POSITIVE' if row['vader_pos'] > row['vader_neg'] else 'NEGATIVE', axis=1
        )

        sentiment_by_year = pd.DataFrame({
            'POSITIVE': data[data['sentiment_vader'] == 'POSITIVE'].groupby('YEA')['RES'].count(),
            'NEGATIVE': data[data['sentiment_vader'] == 'NEGATIVE'].groupby('YEA')['RES'].count(),
        }).fillna(0)

        # Add percentages
        sentiment_by_year['POSITIVE_PERC'] = sentiment_by_year['POSITIVE'] / sentiment_by_year.sum(axis=1)
        sentiment_by_year['NEGATIVE_PERC'] = sentiment_by_year['NEGATIVE'] / sentiment_by_year.sum(axis=1)

        # Reset index for plotting
        sentiment_by_year = sentiment_by_year.reset_index()

        # Data for bar and line plots
        df_g = sentiment_by_year.drop(['POSITIVE_PERC', 'NEGATIVE_PERC'], axis=1)
        df_g['TOTAL'] = df_g.sum(axis=1)

        df_g2 = sentiment_by_year[['YEA', 'POSITIVE_PERC']]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=df_g.YEA,
                    y=df_g.NEGATIVE,
                    name="NEGATIVE",
                    text=df_g.NEGATIVE,
                    textposition="inside",
                    marker=dict(color="salmon")
                ),
                go.Bar(
                    x=df_g.YEA,
                    y=df_g.POSITIVE,
                    name="POSITIVE",
                    text=df_g.POSITIVE,
                    textposition="inside",
                    marker=dict(color="teal")
                ),
                go.Bar(
                    x=df_g.YEA,
                    y=df_g.TOTAL,
                    name="Total",
                    text=df_g.TOTAL,
                    textposition="outside",
                    marker=dict(color="blue")
                ),
            ],
            layout=dict(
                barcornerradius=15,
            ),
        )

        # Overlay line graph for Positive Sentiment Rate
        fig.add_trace(
            go.Scatter(
                x=df_g2.YEA,
                y=df_g2.POSITIVE_PERC * 100,  # Convert to percentage
                mode="lines+markers+text",
                name="Positive Sentiment Rate",
                text=[f"{pr*100:.1f}%" for pr in df_g2["POSITIVE_PERC"]],
                textposition="top center",
                line=dict(color="red", width=2),
                marker=dict(color="red", size=8),
                yaxis="y2"
            )
        )

        # Update layout for dual y-axis
        fig.update_layout(
            title=dict(
                text="Evolution of Positive and Negative Sentiments Over Years",
                x=0.5,
                xanchor="center"
            ),
            xaxis=dict(
                title="Year",
                tickvals=df_g.YEA,
            ),
            yaxis=dict(
                title="Counts",
                side="left"
            ),
            yaxis2=dict(
                title="Positive Sentiment Rate (%)",
                overlaying="y",
                side="right",
                range=[0, 100],
                tickformat=".0f%",
            ),
            barmode="group",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        if prt:
            fig.show()
        if savefig :
            pio.write_html(fig, file="res/Plots/Evolution_of_Sentiments_vader.html", auto_open=False)

    else :
        # Compute sentiment stats by year
        sentiment_by_year = pd.DataFrame({
            'POSITIVE': data[data['sentiment'] == 'POSITIVE'].groupby('YEA')['RES'].count(),
            'NEGATIVE': data[data['sentiment'] == 'NEGATIVE'].groupby('YEA')['RES'].count(),
        })

        # Add percentages
        sentiment_by_year['POSITIVE_PERC'] = sentiment_by_year['POSITIVE'] / sentiment_by_year.sum(axis=1)
        sentiment_by_year['NEGATIVE_PERC'] = sentiment_by_year['NEGATIVE'] / sentiment_by_year.sum(axis=1)

        # Reset index for plotting
        sentiment_by_year = sentiment_by_year.reset_index()

        # Data for bar and line plots
        df_g = sentiment_by_year.drop(['POSITIVE_PERC', 'NEGATIVE_PERC'], axis=1)
        df_g['TOTAL'] = df_g.sum(axis=1)

        df_g2 = sentiment_by_year[['YEA', 'POSITIVE_PERC']]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=df_g.YEA,
                    y=df_g.NEGATIVE,
                    name="NEGATIVE",
                    text=df_g.NEGATIVE,
                    textposition="inside",
                    marker=dict(color="salmon")
                ),
                go.Bar(
                    x=df_g.YEA,
                    y=df_g.POSITIVE,
                    name="POSITIVE",
                    text=df_g.POSITIVE,
                    textposition="inside",
                    marker=dict(color="teal")
                ),
                go.Bar(
                    x=df_g.YEA,
                    y=df_g.TOTAL,
                    name="Total",
                    text=df_g.TOTAL,
                    textposition="outside",
                    marker=dict(color="blue")
                ),
            ],
            layout=dict(
                barcornerradius=15,
            ),
        )

        # Overlay line graph for Positive Sentiment Rate
        fig.add_trace(
            go.Scatter(
                x=df_g2.YEA,
                y=df_g2.POSITIVE_PERC * 100,  # Convert to percentage
                mode="lines+markers+text",
                name="Positive Sentiment Rate",
                text=[f"{pr*100:.1f}%" for pr in df_g2["POSITIVE_PERC"]],
                textposition="top center",
                line=dict(color="red", width=2),
                marker=dict(color="red", size=8),
                yaxis="y2"
            )
        )

        # Update layout for dual y-axis
        fig.update_layout(
            title=dict(
                text="Evolution of Positive and Negative Sentiments Over Years",
                x=0.5,
                xanchor="center"
            ),
            xaxis=dict(
                title="Year",
                tickvals=df_g.YEA,
            ),
            yaxis=dict(
                title="Counts",
                side="left"
            ),
            yaxis2=dict(
                title="Positive Sentiment Rate (%)",
                overlaying="y",
                side="right",
                range=[0, 100],
                tickformat=".0f%",
            ),
            barmode="group",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        if prt:
            fig.show()
        if savefig :
            pio.write_html(fig, file="res/Plots/Evolution_of_Sentiments.html", auto_open=False)
    return None



def visualize_cooperation(prt=False, save_fig=False, path='./res/images/cooperation.webp'):
    df = pre.complete_prepro_w_sa_topics()[0]  # Assuming pre is defined elsewhere

    # Drop rows with NaN values in 'VOT' column
    df = df.dropna(subset=['VOT'])

    required_columns = ['SRC', 'TGT', 'VOT', 'YEA', 'DAT', 'Attempt']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Dataframe must contain columns: {', '.join(required_columns)}")

    # Ensure the 'DAT' column is in datetime format
    df['DAT'] = pd.to_datetime(df['DAT'])

    # Drop the original TIM column, as it's now merged into DAT
    df = df.drop(columns=['TIM'])

    # Sort by DAT to ensure the most recent vote is retained
    df = df.sort_values(by='DAT').drop_duplicates(subset=['SRC', 'TGT', 'Attempt'], keep='last')

    # Get unique years
    years = sorted(df['YEA'].unique())

    # Create a grid of subplots: 3 columns, enough rows to cover all years
    n_cols = 3
    n_rows = -(-len(years) // n_cols)  # Ceiling division to determine number of rows

    # Create a figure and axes for the subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows), sharex=True, sharey=True)

    # Flatten the axes array in case the number of rows and columns don't match exactly
    axes = axes.flatten()

    # Loop through each year and plot the graph on the corresponding subplot
    for idx, year in enumerate(years):
        print(year, '\n')
        ax = axes[idx]
        
        # Filter data for the current year
        year_data = df[df['YEA'] == year]

        # Initialize a graph for this year
        G = nx.Graph()

        # Add nodes explicitly (add only unique SRC values)
        for src in year_data['SRC']:
            G.add_node(src)

        # Group by TGT and VOT, and for each group, create edges between the SRC values
        for (tgt, vot), group in year_data.groupby(['TGT', 'VOT']):
            for i, row1 in group.iterrows():
                for j, row2 in group.iterrows():
                    if i < j:  # Avoid adding duplicate edges
                        color = {1: 'red', -1: 'green', 0: 'blue'}.get(row1['VOT'], 'black')
                        G.add_edge(row1['SRC'], row2['SRC'], color=color)

        # Get the edge colors based on 'VOT'
        edge_colors = [G[u][v]['color'] for u, v in G.edges()]

        # Plot the graph for the current year on the corresponding subplot
        pos = nx.spring_layout(G, k=0.5)  # Adjust the layout for compactness
        nx.draw(G, pos, with_labels=False, node_color='lightblue', edge_color=edge_colors, width=1, node_size=30, font_size=8, ax=ax)  # Reduced node size

        # Set the title for each subplot (year)
        ax.set_title(f"Year {year}")

        # Create a custom legend for the VOT values
        legend_labels = {1: 'VOT=1', -1: 'VOT=-1', 0: 'VOT=0'}
        handles = [Line2D([0], [0], color='red', lw=3, label='VOT=1'),
                   Line2D([0], [0], color='green', lw=3, label='VOT=-1'),
                   Line2D([0], [0], color='blue', lw=3, label='VOT=0')]

        # Add the legend to the plot
        ax.legend(handles=handles, title="Vote (VOT)", loc='upper left')

    # Remove any extra subplots (axes that are not used)
    for i in range(len(years), len(axes)):
        fig.delaxes(axes[i])

    # Adjust the layout for better spacing
    plt.tight_layout()

    if prt:
        plt.show()

    if save_fig:
        fig.savefig(path)

    return None





def plot_clusters(sn_clusters=5, prt=False, save_fig=False, path='./res/Plot/clusters.webp'):
    """
    Plots clusters in a dataset using KMeans clustering, organized by sentiment, topic_x, and topic_y.
    """
    # Preprocess the data
    data = pre.complete_prepro_w_sa_topics()[0]

    # Encode categorical columns
    le_sentiment = LabelEncoder()
    le_topic_x = LabelEncoder()

    data = data.dropna(subset=['topic_y'])

    # Encode sentiment and topic_x as integers
    data['sentiment_encoded'] = le_sentiment.fit_transform(data['sentiment'])
    data['topic_x_encoded'] = le_topic_x.fit_transform(data['topic_x'])

    # One-hot encode topic_y (multiple topics per row)
    mlb = MultiLabelBinarizer()
    topic_y_encoded = mlb.fit_transform(data['topic_y'])

    # Convert one-hot encoding to a DataFrame and merge with original data
    topic_y_df = pd.DataFrame(topic_y_encoded, columns=mlb.classes_, index=data.index)
    data = pd.concat([data, topic_y_df], axis=1)

    # Select features for clustering
    features = ['sentiment_encoded', 'topic_x_encoded'] + list(mlb.classes_)
    X = data[features]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA for 2D visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=sn_clusters, random_state=42)
    data['cluster'] = kmeans.fit_predict(X_pca)

    # Plot clusters
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data['cluster'], cmap='viridis', s=50)
    plt.title('Clusters Based on Sentiment, Topic_x, and Topic_y', fontsize=16)
    plt.xlabel('PCA Component 1', fontsize=14)
    plt.ylabel('PCA Component 2', fontsize=14)
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, linestyle='--', alpha=0.6)

    if prt:
        plt.show()
    if save_fig:
        plt.savefig(path)

    return data[['sentiment', 'topic_x', 'topic_y', 'cluster']]
