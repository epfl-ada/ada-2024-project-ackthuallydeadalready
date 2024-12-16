import src.utils.plotting
from transformers import pipeline
#from transformers import AutoModelForSequenceClassification, AutoTokenizer # Respectively to use a longformer and totokenize,
# not needed yet
import torch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd


# Randomly samples 10% of rows in a dataframe provided, is used to conduct the preliminary sentiment analysis faster
def sampling(df, prop=0.1):
    return df.sample(frac=prop)

# FUnction that performs a sentiment analysis using a hugginface model 
def sa_hug(df,max=512): # Allows user to specify the max number of tokens
    device = 0 if torch.cuda.is_available() else -1 # Uses a gpu if available
    sentiment_analyzer = pipeline("sentiment-analysis", truncation=True, max_length = max, device=device) # Performs the senitment analysis
    results = sentiment_analyzer(df['TXT'].tolist())
    df['sentiment'] = [result['label'] for result in results]
    df['confidence'] = [result['score'] for result in results] #needed? maybe it just uselessly makes the file bigger?
    return df


def sa_vader(df):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = df['TXT'].apply(lambda x:analyzer.polarity_scores(x))
    df[['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound']]=sentiment.apply(lambda x: pd.Series([x['neg'], x['neu'],
                                                                                                      x['pos'], x['compound']]))
    return df


def sa_vader2(df):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = df['TXT'].str.apply(lambda x:analyzer.polarity_scores(x))
    df['vader_neg']=[sent['neg'] for sent in sentiment]
    df['vader_neu']=[sent['neu'] for sent in sentiment]
    df['vader_pos']=[sent['pos'] for sent in sentiment]
    df['compound']=[sent['compound'] for sent in sentiment]
    return df


def sa_vader_test(df):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = df['TXT'].apply(lambda x:analyzer.polarity_scores(x))
    return sentiment