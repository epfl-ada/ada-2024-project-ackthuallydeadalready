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
import seaborn as sns


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


def bar_pass_fail(unique_elections):
    '''
    bar_pass_fail(unique_elections)
    ## Function
    Plots the barplot between Succeded and Failed elections from Unique elections
    ## Variable
    unique_elections : dataframe
    '''
    plt.bar(['RFA Pass','RFA Failed'],unique_elections.RES.value_counts(normalize=True))
    plt.ylabel('Percentage')
    plt.title('Percentage of Passed RFA overall')
    return None


def polarity_scatter(data_vote_polarity):
    '''
    bar_pass_fail(unique_elections)
    ## Function
    Plots the scatter from the data vote polarity
    ## Variable
    data_vote_polarity, dataframe
    '''
    plt.figure(figsize=(8,6))
    plt.scatter(data_vote_polarity.votes, data_vote_polarity.pos_pct)
    return None

def polarity_grid(data_vote_polarity):
    '''
    bar_pass_fail(unique_elections)
    ## Function

    ## Variable
    '''
    sns.FacetGrid(data_vote_polarity, col='YEA').map(sns.regplot, 'votes','pos_pct')
    return None
    
def polarity_lm(data_vote_polarity):
    '''
    bar_pass_fail(unique_elections)
    ## Function
    ## Variable
    '''
    sns.lmplot(x='votes',y='pos_pct', data=data_vote_polarity, hue = 'YEA')
    plt.xlabel("Percentage of Self Employed people [%]")
    plt.ylabel("Income per Capita [$]")
    #plt.ylim([10000,60000])
    #plt.xlim([0,25])

def data_by_tgt(data, print_stats = False):
    '''
    bar_pass_fail(unique_elections)
    ## Function
    ## Variable
    '''
    data_by_tgt = data.groupby(by='TGT').count().sort_values(by = 'SRC', ascending=False)
    data_by_tgt.head(20)

    if print_stats:
        election_mean = data_by_tgt.SRC.mean()
        election_median = data_by_tgt.SRC.median()
        election_mode = data_by_tgt.sort_values('SRC', ascending=True).SRC.quantile(q=0.5)
        election_quant_9 = data_by_tgt.sort_values('SRC', ascending=True).SRC.quantile(q=0.9)

        print('From above, we see that, on average, we have {:.1f} for an election.\
        \nHalf of our elections will have less than {:.0f} votes.\
        \nThe most frequent vote count for an election will be {:.0f} votes\
        \nWe can see that the 90% of the elections have less than {:.0f} votes.\
        '.format(election_mean, election_median, election_mode,election_quant_9))
    else :
        pass

    fig, ax1 = plt.subplots()

    counts, bins, patches = ax1.hist(data_by_tgt.SRC, bins= 200, cumulative = False, color= 'blue', label = 'Histogram')

    #ax1 = plt.hist(x = data_by_tgt.SRC, bins=200, cumulative=False, label='Histogram')
    ax1.set_xlabel('Votes')
    ax1.set_ylabel('Frequency')
    ax1.legend(loc= 'lower right')

    cumulative = np.cumsum(counts)
    cumulative = cumulative / cumulative[-1]

    ax2 = ax1.twinx()

    ax2.plot(bins[:-1], cumulative, color ='red', marker='o', label = 'cumulative distribution')

    ax2.set_ylabel('Cumulative Probability')
    ax2.legend(loc = 'upper right')

    plt.title('Histogram with Cumulative Distribution') 
    plt.show()
    return None


def hist_cummul_vote_pol(data_vote_polarity):
    fig, ax1 = plt.subplots()

    counts, bins, patches = ax1.hist(data_vote_polarity.groupby('TGT')['TXT'].count(), bins= 200, cumulative = False, log = True, alpha = 0.8)

    #ax1 = plt.hist(x = data_by_tgt.SRC, bins=200, cumulative=False, label='Histogram')
    ax1.set_xlabel('Number of Comments')
    ax1.set_ylabel('Frequency (Log Scale)')
    plt.xticks(range(0,1000,100))
    #ax1.legend(loc= 'lower right')

    cumulative = np.cumsum(counts)
    cumulative = cumulative / cumulative[-1]

    ax2 = ax1.twinx()

    ax2.plot(bins[:-1], cumulative, color ='red', alpha= 0.8, marker='o', label = 'cumulative distribution')

    ax2.set_ylabel('Cumulative Probability')
    ax2.legend(loc = 'upper right')

    plt.title('Histogram with Cumulative Distribution') 
    plt.show()

def violins(data_vote_polarity):
    fig, axs = plt.subplots(4,1)

    axs[0].violinplot(data_vote_polarity.groupby('TGT')['TXT'].count()[data_vote_polarity.groupby('TGT')['TXT'].count()<120+1], vert = False)
    axs[0].set_title('90th percentile', fontsize=10)
    axs[0].set_xlim([0,290])

    axs[1].violinplot(data_vote_polarity.groupby('TGT')['TXT'].count()[data_vote_polarity.groupby('TGT')['TXT'].count()<163+1], vert = False)
    axs[1].set_title('95th percentile', fontsize=10)
    axs[1].set_xlim([0,290])

    axs[2].violinplot(data_vote_polarity.groupby('TGT')['TXT'].count()[data_vote_polarity.groupby('TGT')['TXT'].count()<276+1], vert = False)
    axs[2].set_title('99th percentile', fontsize=10)
    axs[2].set_xlim([0,290])

    axs[3].violinplot(data_vote_polarity.groupby('TGT')['TXT'].count()[data_vote_polarity.groupby('TGT')['TXT'].count()>276], vert = False)
    axs[3].set_title('1% Tail', fontsize=10)

    plt.tight_layout()
    plt.show()
    return None


def cumulative_elec_particip(data, print_stats = False):
    user_elec_particip = data.drop_duplicates(subset=['SRC','TGT','YEA','VOT']).groupby('SRC')['SRC'].value_counts().sort_values(ascending=False)

    if print_stats :
        print('90th percentile: {:.0f}'.format(user_elec_particip.quantile(0.90)))
        print('90th percentile: {:.0f}'.format(user_elec_particip.quantile(0.95)))
        print('90th percentile: {:.0f}'.format(user_elec_particip.quantile(0.99)))
    else :
        pass

    fig, ax1 = plt.subplots()

    counts, bins, patches = ax1.hist(user_elec_particip,
                                    bins= 200, cumulative = False, log = True, alpha = 0.8)

    #ax1 = plt.hist(x = data_by_tgt.SRC, bins=200, cumulative=False, label='Histogram')
    ax1.set_xlabel('Number of Elections Participated')
    ax1.set_ylabel('Frequency (Log Scale)')
    #plt.xticks(range(0,1000,100))
    #ax1.legend(loc= 'lower right')

    cumulative = np.cumsum(counts)
    cumulative = cumulative / cumulative[-1]

    ax2 = ax1.twinx()

    ax2.plot(bins[:-1], cumulative, color ='red', alpha= 0.8, marker='o', label = 'cumulative distribution')

    ax2.set_ylabel('Cumulative Probability')
    ax2.legend(loc = 'upper right')
    plt.show()
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



def linear_pred_influ_vot(df, acc_thr=0.95, prt=False, save=False):
    user_accuracy = (
        df.groupby('SRC')
        .apply(lambda x: ((x['RES'] == x['VOT']) | (x['RES']==0) & (x['VOT']==-1)).mean())  # Proportion of correct predictions
        .reset_index(name='accuracy')
    )
    
    election_participation = (
        df.drop_duplicates(subset=['SRC', 'TGT', 'YEA', 'RES'])  # Unique elections
        .groupby('SRC')
        .size()
        .reset_index(name='num_elections')
    )
    
    result = pd.merge(user_accuracy, election_participation, on='SRC')
    influential_users = result[result['accuracy'] >= acc_thr]


    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=result['num_elections'],
        y=result['accuracy'],
        mode='markers',
        marker=dict(color='blue', opacity=0.6),
        name='User Accuracy vs Elections'
    ))
    
    median_elections = result['num_elections'].median()
    fig.add_trace(go.Scatter(
        x=[median_elections, median_elections],
        y=[0, 1],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Median Elections'
    ))

    fig.add_trace(go.Scatter(
        x=[result['num_elections'].min(), result['num_elections'].max()],
        y=[acc_thr, acc_thr],
        mode='lines',
        line=dict(color='green', dash='dash'),
        name=f'Accuracy Threshold ({acc_thr * 100}%)',
        hovertemplate=(
            'SRC: %{text}<br>' +
            'Accuracy: %{y:.2f}<br>' +
            'Num Elections: %{x}<br>' +
            '<extra></extra>'
        ),
        text=result['SRC'], 
    ))

    fig.update_layout(
        title='Accuracy vs Number of Elections Participated in',
        xaxis_title='Number of Elections Participated in',
        yaxis_title='Accuracy (Proportion of Correct Predictions)',
        showlegend=True, template='plotly_white'
    )
    if prt:
        fig.show()
    if save:
        fig.write_html('res/Plots/accuracy_vs_participation.html')
    return None


def pass_rate_once_v_mult(sing_mult_stats, prt = False, savefig = False):
    df_g =sing_mult_stats.drop(['SING_PERC','MULT_PERC'],axis =1).transpose().reset_index().rename(columns={'index':'type', -1:'neg', 1:'pos'})
    df_g2 =sing_mult_stats.drop(['SING','MULT'],axis =1).transpose().reset_index().rename(columns={'index':'type', -1:'neg', 1:'pos'})
    df_g2


    fig = go.Figure(
        data=[
            go.Bar(x=df_g.type, y=df_g.neg, name="Losses", text=df_g.neg, textposition='inside'),
            go.Bar(x=df_g.type, y=df_g.pos, name="Wins", text=df_g.pos, textposition=['inside','outside']),
            go.Bar(x=df_g.type, y=df_g.Total, name="Total", text=df_g.Total, textposition='inside'),
        ],
        layout=dict(
            barcornerradius=15,
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=df_g.type,  # Use the same x values as the "Wins" bar
            y=[pr * 100 for pr in df_g2["pos"]],  # Convert pass rate to percentage
            mode="lines+markers+text",  # Line graph with markers and text
            name="Pass Rate",
            text=[f"{pr*100:.1f}%" for pr in df_g2["pos"]],  # Add pass rate as percentage text
            textposition="top center",
            line=dict(color="red", width=2),
            marker=dict(color="red", size=8),
            yaxis="y2",  # Associate this trace with the secondary y-axis
        )
    )

    fig.update_layout(
        title=dict(
            text="Candidate Pass Rates: One Time vs Multiple Times",  # Chart title
            x=0.5,  # Center the title
            xanchor="center",  # Anchor the title at the center
        ),
        xaxis=dict(
            title="Times Candidate Participated",
            tickvals=["SING", "MULT"],  # Define custom labels
            ticktext=["Candidate Only Once", "Candidate Multiple Times"],
        ),
        yaxis=dict(title="Counts", side="left"),
        yaxis2=dict(
            title="Pass Rate (%)",
            overlaying="y",
            side="right",
            range=[0, 100],
            tickformat=".0f%",
        ),
        barmode="group",
        legend=dict(
            title="Legend",
            x=1.15,
            y=0.6,
            xanchor="right",
            yanchor="middle",
        ),
    )

    if prt:
        fig.show()
    if savefig :
        pio.write_html(fig, file="res/Plots/pass_rates_once_v_multiple.html", auto_open=False)
    return None



def passrate_byYear_plot(passrate_by_year, prt = False, savefig = False):
    df_g = passrate_by_year.drop(['WIN_PERC','LOSS_PERC'],axis =1)
    df_g['TOTAL'] = df_g.sum(axis=1)
    df_g = df_g.reset_index()

    df_g2 = passrate_by_year.drop(['WIN','LOSS'],axis =1).reset_index().rename(columns={'index':'type'})

    fig = go.Figure(
        data=[
            go.Bar(x=df_g.YEA, y=df_g.LOSS, name="LOSS", text=df_g.LOSS, textposition='inside'),
            go.Bar(x=df_g.YEA, y=df_g.WIN, name="WIN", text=df_g.WIN, textposition=['inside','outside']),
            go.Bar(x=df_g.YEA, y=df_g.TOTAL, name="Total", text=df_g.TOTAL, textposition='inside'),
        ],
        layout=dict(
            barcornerradius=15,
        ),
    )

    fig.update_traces(
        textfont=dict(color="black"),  
        selector=dict(type="bar")  
    )

    fig.add_trace(
        go.Scatter(
            x=df_g.YEA,  
            y=[pr * 100 for pr in df_g2["WIN_PERC"]],  # Convert pass rate to percentage
            mode="lines+markers+text", 
            name="Pass Rate",
            text=[f"{pr*100:.1f}%" for pr in df_g2["WIN_PERC"]],  
            textposition=['bottom center',"top center","top center","top center","top center","top center","top center","top center","top center","top center","top center"],
            line=dict(color="red", width=2),
            marker=dict(color="red", size=8),
            yaxis="y2",  
        )
    )

    fig.update_layout(
        title=dict(
            text="Elections and Win Rates Across The Years",  
            x=0.5,  
            xanchor="center",  
        ),
        xaxis=dict(
            title="Year",
            tickvals=df_g.YEA,
            
        ),
        yaxis=dict(title="Counts", side="left"),
        yaxis2=dict(
            title="Pass Rate (%)",
            overlaying="y",
            side="right",
            range=[0, 100],
            tickformat=".0f%",
        ),
        barmode="group",
        legend=dict(
            title="Legend",
            x=1.15,
            y=0.6,
            xanchor="right",
            yanchor="middle",
        ),
    )


    if prt:
        fig.show()
    if savefig :
        pio.write_html(fig, file="res/Plots/Evolution_of_Pass_Rates_and_votes.html", auto_open=False)
    return None


def outcome_over_time(df1, prt = False, savefig = False):
    df_g = pd.DataFrame(df1.groupby('RES')[['YEA','RES','TOT']].value_counts())
    df_g = df_g.reset_index()
    df_g = df_g.drop('count', axis=1)

    year = 2003
    rows = 4
    cols = 3

    # Create subplot layout with titles for each subplot
    subplot_titles = [f"Year {year + i}" for i in range(12)]
    subplot_titles[-1] = "" 

    fig = make_subplots(rows = rows, cols = cols, subplot_titles=subplot_titles)

    color_pairs = [
        ("rgba(135, 206, 250, 0.8)", "rgba(255, 165, 0, 0.8)"),  # Light blue & orange
        ("rgba(147, 112, 219, 0.8)", "rgba(0, 255, 127, 0.8)"),  # Purple & teal
        ("rgba(173, 255, 47, 0.8)", "rgba(255, 105, 180, 0.8)"),  # Green & pink
    ]


    for idx in range(0,11,1):

        row = (idx // cols) + 1  # Integer division to determine row
        col = (idx % cols) + 1   # Modulo to determine column

            # Alternate colors based on the index
        loss_color, win_color = color_pairs[idx % len(color_pairs)]

        # Add Histograms for RES=1 and RES=-1
        fig.add_trace(
            go.Histogram(x=df_g[(df_g['YEA'] == year + idx) & (df_g['RES'] == 1)]['TOT'], 
                        name='Win',
                        marker_color= win_color,
                        showlegend =False
                        ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Histogram(x=df_g[(df_g['YEA'] == year + idx) & (df_g['RES'] == -1)]['TOT'], 
                        name='Loss',
                        marker_color = loss_color,
                        showlegend =False
                        ),
            row=row,
            col=col,
        )


    
    fig.update_layout(
        height=1000,       
        width=1600,      
        
    )

    fig.update_xaxes(title_text="Total Count", row=rows, col=2)  
    fig.update_yaxes(title_text="Frequency", row=2, col=1)      

    common_xrange = [0, 350]  
    fig.update_xaxes(range=common_xrange)

    common_yrange = [0, 50]
    fig.update_yaxes(range=common_yrange)

    fig.update_xaxes(visible=False, row=rows, col=cols)
    fig.update_yaxes(visible=False, row=rows, col=cols)

    fig.update_layout(barmode='overlay', title =dict(
            text="Distribution of Votes by Election Outcome over the Years",
            x=0.5,  
            xanchor="center", 
        )
    )
    fig.update_traces(opacity=0.8)

    if prt:
        fig.show()
    if savefig :
        pio.write_html(fig, file="Plots/dist_votes_by_elec_outcome_over_years.html", auto_open=False)
    return None


def voting_behaviors(df1,passrate_by_year, prt = False, savefig = False):
    df_g = df1.drop_duplicates('elec_id')[['YEA','TOT','POS','ABS','NEG']]
    df_g = df_g.reset_index()
    df_g = df_g.drop('index', axis=1)
    df_g = df_g.groupby('YEA').sum().reset_index()

    df_g['POS_pct'] = (df_g['POS'] / df_g['TOT']) * 100
    df_g['ABS_pct'] = (df_g['ABS'] / df_g['TOT']) * 100
    df_g['NEG_pct'] = (df_g['NEG'] / df_g['TOT']) * 100


    df_g2 = passrate_by_year.drop(['WIN','LOSS'],axis =1).reset_index().rename(columns={'index':'type'})

    df_g3 = df1.drop_duplicates('elec_id').groupby('YEA')[['elec_id']].count().rename(columns={'elec_id':'no_elec'})
    df_g3 = df_g3.reset_index()
    df_g3['VOT_PER_ELEC'] = df_g['TOT']/df_g3['no_elec']

    df_melted = df_g.melt(
        id_vars=['YEA', 'TOT'], 
        value_vars=['POS', 'ABS', 'NEG'], 
        var_name='Type', 
        value_name='Value'
    )

    df_melted['Percentage'] = df_melted.apply(
        lambda row: (row['Value'] / df_g.loc[df_g['YEA'] == row['YEA'], 'TOT'].values[0]) * 100, axis=1
    )

    fig = go.Figure()

    # Add the bar chart with percentage text
    for type_name in df_melted['Type'].unique():
        filtered_df = df_melted[df_melted['Type'] == type_name]
        fig.add_trace(
            go.Bar(
                x=filtered_df['YEA'],
                y=filtered_df['Value'],
                name=type_name,
                text=[f"{p:.1f}%" for p in filtered_df['Percentage']],
                textposition='inside'
            ),
            
        )


    fig.update_traces(
        textfont=dict(color="white"),  
        selector=dict(type="bar") 
    )

    # Add the scatter plot for pass rates
    fig.add_trace(
        go.Scatter(
            x=df_g['YEA'],  # Use the same x values
            y=[pr * 100 for pr in df_g2["WIN_PERC"]], 
            mode="lines+markers+text",  
            name="Pass Rate",
            text=[f"{pr*100:.1f}%" for pr in df_g2["WIN_PERC"]],  
            textposition='bottom center',
            line=dict(color="red", width=2),
            marker=dict(color="red", size=8),
            yaxis="y2"  
        )
    )

    fig.update_traces(
        textfont=dict(color="red"), 
        selector=dict(type="scatter")  
    )

    # Configure layout
    fig.update_layout(
        barmode='stack',
        title=dict(
            text="Voting Behavior and Pass Rates over a Decade",  
            x=0.5,  
            xanchor="center",  
        ),
        xaxis=dict(
            title="Year",
            tickmode='linear',  
            tick0=2003,        
            dtick=1            
        ),
        yaxis=dict(title="Total Count"),
        yaxis2=dict(
            title="Pass Rate (%)",
            overlaying="y",
            side="right",  
            range=[0, 100]  
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=800
    )

    fig.add_trace(
        go.Scatter(
            x=df_g3['YEA'],  
            y=[50000] * len(df_g3),  
            mode="markers+text",  
            name="Votes Per Election",
            text=[f"{v:.0f}" for v in df_g3['VOT_PER_ELEC']],  
            textposition="top center",  
            marker=dict(
                size=df_g3['VOT_PER_ELEC'] *1, 
                color="blue",
                opacity=0.7
            ),
            hoverinfo="text+x", 
            hovertext=[f"Year: {year}<br>Votes Per Election: {votes:.0f}" for year, votes in zip(df_g3['YEA'], df_g3['VOT_PER_ELEC'])]
        )
    )

    # Update bubbles 
    fig.update_layout(
        title=dict(
            text="Voting Behavior, Pass Rates, and Votes Per Election",
            x=0.5,
            xanchor="center"
        ),
        xaxis=dict(
            title="Year",
            tickmode='array', 
            tickvals=[2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013],  
            ticktext=["2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013"]  
        ),
        yaxis=dict(
            title="Total Count"
        ),
        yaxis2=dict(
            title="Pass Rate (%)",
            overlaying="y",
            side="right",
            range=[0, 120],  
            tickvals=[0, 20, 40, 60, 80, 100],  
            ticktext=["0%", "20%", "40%", "60%", "80%", "100%"]  
        ),
        height=1000,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        barcornerradius=10 
    )

    if prt:
        fig.show()
    if savefig :
        pio.write_html(fig, file="Plots/voting_beh_passrates_vot_per_elec.html", auto_open=False)
    return None



def plot_network(df, prt = False, savefig = False, path = './res/Plots/connexion.webp'):
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

    pos = nx.spring_layout(G) 

    node_x = [pos[node][0] for node in G.nodes]
    node_y = [pos[node][1] for node in G.nodes]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(size=10, color='gray'),
    )

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

        sentiment_stats_vader['Total'] = sentiment_stats_vader.sum(axis=1)
        sentiment_stats_vader['Pass Rate'] = sentiment_stats_vader[1] / sentiment_stats_vader['Total']

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

        fig.add_trace(
            go.Scatter(
                x=df_g2['sentiment_vader'],
                y=df_g2['Pass Rate'] * 100,  
                mode="lines+markers+text",
                name="Pass Rate",
                text=[f"{pr * 100:.1f}%" for pr in df_g2["Pass Rate"]],
                textposition="top center",
                line=dict(color="red", width=2),
                marker=dict(color="red", size=8),
                yaxis="y2"
            )
        )

        fig.update_layout(
            title=dict(
                text="Success Rate vs Voter Sentiment using Vader",
                x=0.5,
                xanchor="center"
            ),
            xaxis=dict(
                title="Voter Sentiment"
            ),
            yaxis=dict(
                title="Comments Counts",
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

        fig.add_trace(
            go.Scatter(
                x=df_g2['sentiment'],
                y=df_g2['Pass Rate'] * 100, 
                mode="lines+markers+text",
                name="Pass Rate",
                text=[f"{pr * 100:.1f}%" for pr in df_g2["Pass Rate"]],
                textposition="top center",
                line=dict(color="red", width=2),
                marker=dict(color="red", size=8),
                yaxis="y2"
            )
        )

        fig.update_layout(
            title=dict(
                text="Success Rate vs Voter Sentiment using HuggingFace",
                x=0.5,
                xanchor="center"
            ),
            xaxis=dict(
                title="Voter Sentiment"
            ),
            yaxis=dict(
                title="Comment Counts",
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

        sentiment_by_year['POSITIVE_PERC'] = sentiment_by_year['POSITIVE'] / sentiment_by_year.sum(axis=1)
        sentiment_by_year['NEGATIVE_PERC'] = sentiment_by_year['NEGATIVE'] / sentiment_by_year.sum(axis=1)

        sentiment_by_year = sentiment_by_year.reset_index()

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

        fig.update_layout(
            title=dict(
                text="Evolution of Positive and Negative Sentiments Over Years using Vader",
                x=0.5,
                xanchor="center"
            ),
            xaxis=dict(
                title="Year",
                tickvals=df_g.YEA,
            ),
            yaxis=dict(
                title="Comment Counts",
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
        sentiment_by_year = pd.DataFrame({
            'POSITIVE': data[data['sentiment'] == 'POSITIVE'].groupby('YEA')['RES'].count(),
            'NEGATIVE': data[data['sentiment'] == 'NEGATIVE'].groupby('YEA')['RES'].count(),
        })

        sentiment_by_year['POSITIVE_PERC'] = sentiment_by_year['POSITIVE'] / sentiment_by_year.sum(axis=1)
        sentiment_by_year['NEGATIVE_PERC'] = sentiment_by_year['NEGATIVE'] / sentiment_by_year.sum(axis=1)

        sentiment_by_year = sentiment_by_year.reset_index()

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

        fig.add_trace(
            go.Scatter(
                x=df_g2.YEA,
                y=df_g2.POSITIVE_PERC * 100, 
                mode="lines+markers+text",
                name="Positive Sentiment Rate",
                text=[f"{pr*100:.1f}%" for pr in df_g2["POSITIVE_PERC"]],
                textposition="top center",
                line=dict(color="red", width=2),
                marker=dict(color="red", size=8),
                yaxis="y2"
            )
        )

        fig.update_layout(
            title=dict(
                text="Evolution of Positive and Negative Sentiments Over Years using HuggingFace",
                x=0.5,
                xanchor="center"
            ),
            xaxis=dict(
                title="Year",
                tickvals=df_g.YEA,
            ),
            yaxis=dict(
                title="Comment Counts",
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



def visualize_cooperation(prt=False, save_fig=False, path='./res/Plots/cooperation.webp'):
    df = pre.complete_prepro_w_sa_topics()[0]  

    df = df.dropna(subset=['VOT'])

    required_columns = ['SRC', 'TGT', 'VOT', 'YEA', 'DAT', 'Attempt']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Dataframe must contain columns: {', '.join(required_columns)}")

    df['DAT'] = pd.to_datetime(df['DAT'])

    df = df.drop(columns=['TIM'])

    df = df.sort_values(by='DAT').drop_duplicates(subset=['SRC', 'TGT', 'Attempt'], keep='last')

    years = sorted(df['YEA'].unique())


    n_cols = 3
    n_rows = -(-len(years) // n_cols)  

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows), sharex=True, sharey=True)

    axes = axes.flatten()

    for idx, year in enumerate(years):
        print(year, '\n')
        ax = axes[idx]
        
        year_data = df[df['YEA'] == year]

        G = nx.Graph()

        for src in year_data['SRC']:
            G.add_node(src)

        for (tgt, vot), group in year_data.groupby(['TGT', 'VOT']):
            for i, row1 in group.iterrows():
                for j, row2 in group.iterrows():
                    if i < j:  # Avoid adding duplicate edges
                        color = {1: 'red', -1: 'green', 0: 'blue'}.get(row1['VOT'], 'black')
                        G.add_edge(row1['SRC'], row2['SRC'], color=color)

        edge_colors = [G[u][v]['color'] for u, v in G.edges()]

        pos = nx.spring_layout(G, k=0.5)  
        nx.draw(G, pos, with_labels=False, node_color='none', edge_color=edge_colors, width=1, node_size=10, 
                font_size=8, ax=ax, node_shape='o', linewidths=2, edgecolors='black')  # Reduced node size

        ax.set_title(f"Year {year}")

        legend_labels = {1: 'VOT=1', -1: 'VOT=-1', 0: 'VOT=0'}
        handles = [Line2D([0], [0], color='red', lw=3, label='VOT=1'),
                   Line2D([0], [0], color='green', lw=3, label='VOT=-1'),
                   Line2D([0], [0], color='blue', lw=3, label='VOT=0')]

        ax.legend(handles=handles, title="Vote (VOT)", loc='upper left')

    for i in range(len(years), len(axes)):
        fig.delaxes(axes[i])

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

    features = ['sentiment_encoded', 'topic_x_encoded'] + list(mlb.classes_)
    X = data[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

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

def plot_topics_pass(data,comment=True,prt=False,savefig=False):
    if comment:
        topic_frequencies = data['topic_x'].value_counts()

        # Select the top 10 most frequent topics
        top_10_topics = topic_frequencies.head(10).index

        # Filter the data for top 10 topics
        filtered_data = data[data['topic_x'].isin(top_10_topics)]

        # Aggregate counts of passing and failing votes by topic
        topic_stats = filtered_data.groupby(['topic_x', 'RES']).size().unstack(fill_value=0).rename(columns={1: 'Pass', -1: 'Fail'})

        # Add totals and normalize (optional)
        topic_stats['Total'] = topic_stats.sum(axis=1)
        topic_stats['Pass_Perc'] = topic_stats['Pass'] / topic_stats['Total']
        topic_stats['Fail_Perc'] = topic_stats['Fail'] / topic_stats['Total']

        # Reset index and sort by total frequency
        topic_stats = topic_stats.reset_index().sort_values(by='Total', ascending=False)

        # Create a grouped bar chart
        fig = go.Figure(
            data=[
                go.Bar(name="Pass", x=topic_stats['topic_x'], y=topic_stats['Pass'], marker_color='teal'),
                go.Bar(name="Fail", x=topic_stats['topic_x'], y=topic_stats['Fail'], marker_color='salmon'),
            ],
            layout=dict(barcornerradius=15)
        )

        fig.update_layout(
            title="Vote Outcomes for Top 10 Topics (Ordered by Number of Occurences) for Comments",
            xaxis_title="Topics",
            yaxis_title="Count",
            barmode='group',  # Group bars side-by-side
            legend_title="Outcome",
        )

        if prt:
            fig.show()
        if savefig:
            fig.write_html('./res/Plots/topics_pass_comments.html')
    else :
        data['topic_y'] = data['topic_y'].str.replace(r"[{}']", "", regex=True)

        data['topic_y'] = data['topic_y'].str.split(',')
        data_exploded = data.explode('topic_y')

        topic_frequencies = data_exploded['topic_y'].value_counts()

        top_10_topics = topic_frequencies.head(10).index

        filtered_data = data_exploded[data_exploded['topic_y'].isin(top_10_topics)]

        topic_stats = filtered_data.groupby(['topic_y', 'RES']).size().unstack(fill_value=0).rename(columns={1: 'Pass', -1: 'Fail'})

        topic_stats['Total'] = topic_stats.sum(axis=1)
        topic_stats['Pass_Perc'] = topic_stats['Pass'] / topic_stats['Total']
        topic_stats['Fail_Perc'] = topic_stats['Fail'] / topic_stats['Total']

        topic_stats = topic_stats.reset_index().sort_values(by='Total', ascending=False)

        fig = go.Figure(
            data=[
                go.Bar(name="Pass", x=topic_stats['topic_y'], y=topic_stats['Pass'], marker_color='teal'),
                go.Bar(name="Fail", x=topic_stats['topic_y'], y=topic_stats['Fail'], marker_color='salmon'),
            ],
            layout=dict(barcornerradius=15)
        )

        fig.update_layout(
            title="Vote Outcomes for Top 10 Topics (Ordered by Number of Occurences) for Discussions",
            xaxis_title="Topics",
            yaxis_title="Count",
            barmode='group', 
            legend_title="Outcome",
        )

        if prt:
            fig.show()
        if savefig:
            fig.write_html('./res/Plots/topics_pass_discussions.html')
    return None