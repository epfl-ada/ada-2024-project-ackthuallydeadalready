import pandas as pd
import datetime as dt
from datetime import datetime
import calendar 
import plotly.express as px
import plotly.graph_objects as go

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

def dates_prep(data):
    '''
    dates_prep(data[, optional_param])
    ## Function
    This function sparse the DAT column in a proper format and insert YYYY-MM-DD in the previous DAT column, adds a column for the month (MON) and for the time of the vote (TIM).
    Specify optional parameters here.
    ## Variables
    - data : dataframe to process, containing a column YEA with the year and a column DAT formatted as HH:MM, DD Month YYYY
    '''
    data['YEA'] = data['YEA'].astype(int)
    data['DAT'] = data['DAT'].astype(str)
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
    return data


def plot_cooperation_graph(df):
    # Ensure that 'YEA' is in the correct format (integer or datetime) for grouping
    #     
    # Create an empty list to hold the year-by-year matrices of cooperation
    cooperation_matrices = {}
    
    # Loop over unique years
    for year in df['YEA'].unique():
        # Filter data for the current year
        year_data = df[df['YEA'] == year]
        
        # Create a matrix where rows and columns are SRC (users)
        users = year_data['SRC'].unique()
        cooperation_matrix = pd.DataFrame(0, index=users, columns=users)
        
        # Check cooperation between users based on the same TGT and VOT
        for target in year_data['TGT'].unique():
            target_data = year_data[year_data['TGT'] == target]
            
            # Iterate through all pairs of users for the current target
            for i, user1 in enumerate(target_data['SRC'].unique()):
                for j, user2 in enumerate(target_data['SRC'].unique()):
                    if user1 != user2:
                        # Check if both users voted the same way
                        vote1 = target_data[target_data['SRC'] == user1]['VOT'].values[0]
                        vote2 = target_data[target_data['SRC'] == user2]['VOT'].values[0]
                        if vote1 == vote2:  # Same vote, mark as cooperation
                            cooperation_matrix.at[user1, user2] += 1
                            cooperation_matrix.at[user2, user1] += 1  # Symmetric cooperation
        
        # Store the cooperation matrix for the year
        cooperation_matrices[year] = cooperation_matrix
    
    # Now create the sliding plot with Plotly heatmap
    fig = go.Figure()
    
    for year, matrix in cooperation_matrices.items():
        fig.add_trace(go.Heatmap(
            z=matrix.values,
            x=matrix.columns,
            y=matrix.index,
            colorscale='Viridis',
            colorbar=dict(title='Cooperation'),
            showscale=True,
            zmax=matrix.values.max(),
            zmin=0,
            name=f"Year {year}",
            opacity=0.7
        ))
    
    fig.update_layout(
        title='Year-by-Year User Cooperation',
        xaxis_title='Users',
        yaxis_title='Users',
        updatemenus=[dict(
            type="buttons", 
            showactive=False, 
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None, dict(frame=dict(duration=1000, redraw=True), fromcurrent=True)])]
        )],
        sliders=[dict(
            currentvalue=dict(prefix="Year: ", font=dict(size=20)),
            steps=[dict(label=str(year), method="animate", args=[[f"Year {year}"], dict(frame=dict(duration=1000, redraw=True), fromcurrent=True)]) for year in cooperation_matrices.keys()])
            ])
    fig.show()

