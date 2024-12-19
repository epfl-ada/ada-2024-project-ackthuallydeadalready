import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor

def linear_pred_influ_vot(df, acc_thr=0.95, save=False):
    user_accuracy = (
        df.groupby('SRC')
        .apply(lambda x: ((x['RES'] == x['VOT']) | (x['RES']==0) & (x['VOT']==-1)).mean())  # Proportion of correct predictions
        .reset_index(name='accuracy')
    )
    
    # Filter users with at least 95% accuracy
    election_participation = (
        df.drop_duplicates(subset=['SRC', 'TGT', 'YEA', 'RES'])  # Unique elections
        .groupby('SRC')
        .size()
        .reset_index(name='num_elections')
    )
    
    # Merge the two DataFrames
    result = pd.merge(user_accuracy, election_participation, on='SRC')
    influential_users = result[result['accuracy'] >= acc_thr]


    fig = go.Figure()

    # Add scatter plot for all users
    fig.add_trace(go.Scatter(
        x=result['num_elections'],
        y=result['accuracy'],
        mode='markers',
        marker=dict(color='blue', opacity=0.6),
        name='User Accuracy vs Elections'
    ))
    
    # Add a vertical line for the median number of elections
    median_elections = result['num_elections'].median()
    fig.add_trace(go.Scatter(
        x=[median_elections, median_elections],
        y=[0, 1],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Median Elections'
    ))

    # Add a horizontal line for the accuracy threshold
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
        text=result['SRC'],  # Pass SRC column for hover text
    ))

    # Update the layout of the plot
    fig.update_layout(
        title='Accuracy vs Number of Elections Participated in',
        xaxis_title='Number of Elections Participated in',
        yaxis_title='Accuracy (Proportion of Correct Predictions)',
        showlegend=True, template='plotly_white'
    )
    if save:
        fig.write_html('Plots/accuracy_vs_participation.html')
    fig.show()
    return influential_users




def preprocess_and_predict_lin(df, smote=False):
    # Drop rows with NaN values
    df = df.dropna(subset=['sentiment', 'vader_neg', 'vader_neu', 'vader_pos', 'vader_compound', 'topic_x', 'topic_y', 'RES'])
    
    # Sentiment binary encoding
    df['sentiment_binary'] = df['sentiment'].apply(lambda x: 1 if x == 'POSITIVE' else 0)

    # Frequency encoding for topic_x
    freq_map = df['topic_x'].value_counts(normalize=True).to_dict()
    df['topic_x_encoded'] = df['topic_x'].map(freq_map)

    # Sparse encoding for topic_y
    df['topic_y_str'] = df['topic_y'].apply(lambda x: ' '.join(x) if isinstance(x, set) else '')
    vectorizer = CountVectorizer()
    topic_y_sparse = vectorizer.fit_transform(df['topic_y_str'])
    topic_y_df = pd.DataFrame(topic_y_sparse.toarray(), index=df.index)

    # Combine features
    numerical_features = df[['sentiment_binary', 'vader_neg', 'vader_neu', 'vader_pos', 'vader_compound', 'topic_x_encoded']]
    X = pd.concat([numerical_features, topic_y_df], axis=1)
    y = df['RES']

    X.columns = X.columns.astype(str)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Remove outliers (z-score) only from training data
    z_train = zscore(X_train[['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound']])
    X_train = X_train[(np.abs(z_train) < 3).all(axis=1)]
    y_train = y_train[X_train.index]  # Adjust y_train to match X_train

    # Scale numerical features (after removing outliers)
    scaler = StandardScaler()
    numerical_features_scaled_train = scaler.fit_transform(X_train[['sentiment_binary', 'vader_neg', 'vader_neu', 'vader_pos', 'vader_compound', 'topic_x_encoded']])
    X_train[['sentiment_binary', 'vader_neg', 'vader_neu', 'vader_pos', 'vader_compound', 'topic_x_encoded']] = numerical_features_scaled_train

    # Scale numerical features for the test set (using the training scaler)
    numerical_features_scaled_test = scaler.transform(X_test[['sentiment_binary', 'vader_neg', 'vader_neu', 'vader_pos', 'vader_compound', 'topic_x_encoded']])
    X_test[['sentiment_binary', 'vader_neg', 'vader_neu', 'vader_pos', 'vader_compound', 'topic_x_encoded']] = numerical_features_scaled_test

    if smote:
        # Apply SMOTE to balance the training set
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

        # Initialize and train MLPRegressor model
        model = MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
        model.fit(X_train_smote, y_train_smote)
    else:
        # Initialize and train MLPRegressor model
        model = MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
        model.fit(X_train, y_train)


    # Predictions
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Performance Metrics:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

    return model, rmse, mae, r2


def preprocess_and_predict_test(df, smote=False):
    # Drop rows with NaN values
    df = df.dropna(subset=['sentiment', 'vader_neg', 'vader_neu', 'vader_pos', 'vader_compound', 'topic_x', 'topic_y', 'RES'])
    
    # Sentiment binary encoding
    df['sentiment_binary'] = df['sentiment'].apply(lambda x: 1 if x == 'POSITIVE' else 0)

    # Frequency encoding for topic_x
    freq_map = df['topic_x'].value_counts(normalize=True).to_dict()
    df['topic_x_encoded'] = df['topic_x'].map(freq_map)

    # Sparse encoding for topic_y
    df['topic_y_str'] = df['topic_y'].apply(lambda x: ' '.join(x) if isinstance(x, set) else '')
    vectorizer = CountVectorizer()
    topic_y_sparse = vectorizer.fit_transform(df['topic_y_str'])
    topic_y_df = pd.DataFrame(topic_y_sparse.toarray(), index=df.index)

    # Combine features
    numerical_features = df[['sentiment_binary', 'vader_neg', 'vader_neu', 'vader_pos', 'vader_compound', 'topic_x_encoded']]
    X = pd.concat([numerical_features, topic_y_df], axis=1)
    y = df['RES']

    X.columns = X.columns.astype(str)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Remove outliers (z-score) only from training data
    z_train = zscore(X_train[['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound']])
    X_train = X_train[(np.abs(z_train) < 3).all(axis=1)]
    y_train = y_train[X_train.index]  # Adjust y_train to match X_train

    # Scale numerical features (after removing outliers)
    scaler = StandardScaler()
    numerical_features_scaled_train = scaler.fit_transform(X_train[['sentiment_binary', 'vader_neg', 'vader_neu', 'vader_pos', 'vader_compound', 'topic_x_encoded']])
    X_train[['sentiment_binary', 'vader_neg', 'vader_neu', 'vader_pos', 'vader_compound', 'topic_x_encoded']] = numerical_features_scaled_train

    # Scale numerical features for the test set (using the training scaler)
    numerical_features_scaled_test = scaler.transform(X_test[['sentiment_binary', 'vader_neg', 'vader_neu', 'vader_pos', 'vader_compound', 'topic_x_encoded']])
    X_test[['sentiment_binary', 'vader_neg', 'vader_neu', 'vader_pos', 'vader_compound', 'topic_x_encoded']] = numerical_features_scaled_test

    if smote:
        # Apply SMOTE to balance the training set
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

        # Initialize and train MLPRegressor model
        model = MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
        model.fit(X_train_smote, y_train_smote)
    else:
        # Initialize and train MLPRegressor model
        model = MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
        model.fit(X_train, y_train)


    # Predictions
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Performance Metrics:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

    return model, rmse, mae, r2
