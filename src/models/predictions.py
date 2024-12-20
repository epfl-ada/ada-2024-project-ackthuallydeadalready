import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression


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
        fig.write_html('res/Plots/accuracy_vs_participation.html')
    fig.show()
    return influential_users


def preprocess_and_predict(df, smote=False, model_type="lin_reg"):
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
        X_train, y_train = smote.fit_resample(X_train, y_train)

    
    if (model_type=='small_nn'):
        model = MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
        model.fit(X_train, y_train)

    elif (model_type=='lin_reg'):
        model = LinearRegression()
        model.fit(X_train, y_train)

    elif model_type == "xgboost":
        # XGBoost with small grid search and cross-validation
        param_grid = {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
        xgb = XGBRegressor(random_state=42, verbosity=0)
        grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        print(f"Best Parameters for XGBoost: {grid_search.best_params_}")

    elif model_type == "logistic":
        # Logistic regression for interpretability
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)

    else:
        raise ValueError("Invalid model_type. Choose 'lin_reg', 'small_nn', 'xgboost' or 'logistic'.")

    # Predictions
    y_pred = model.predict(X_test)

    # Display performance metrics
    if model_type in ["lin_reg", "small_nn", "xgboost"]:
        # Regression metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print("Regression Metrics:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        return model, rmse, mae, r2

    elif model_type == "logistic":
        # Classification metrics
        y_pred_binary = (y_pred > 0.5).astype(int)  # Assuming threshold 0.5 for classification
        accuracy = accuracy_score(y_test, y_pred_binary)
        report = classification_report(y_test, y_pred_binary)
        print("Classification Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(report)
        return model, accuracy, report