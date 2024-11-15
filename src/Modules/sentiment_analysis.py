import src.Modules.plotting
from transformers import pipeline
#from transformers import AutoModelForSequenceClassification, AutoTokenizer # Respectively to use a longformer and totokenize,
# not needed yet
import torch

# Randomly samples 10% of rows in a dataframe provided, is used to conduct the preliminary sentiment analysis faster
def sampling(df, prop=0.1):
    return df.sample(frac=prop)

# FUnction that performs a sentiment analysis using a hugginface model 
def sa_hug(df,max=512): # Allows user to specify the max number of tokens
    device = 0 if torch.cuda.is_available() else -1 # Uses a gpu if available
    sentiment_analyzer = pipeline("sentiment-analysis", truncation=True, max_length = max, device=device) # Performs the senitment analysis
    results = sentiment_analyzer(df['TXT'].tolist())
    df['sentiment'] = [result['label'] for result in results]
    df['confidence'] = [result['score'] for result in results]
    return df




