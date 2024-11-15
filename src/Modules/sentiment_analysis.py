import src.Modules.plotting
from transformers import AutoTokenizer, pipeline
#from transformers import AutoModelForSequenceClassification #to use a longformer, not needed yet
import torch

def sampling(df, prop=0.1):
    return df.sample(frac=prop)

def sa_hug(df,max=512):
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    sentiment_analyzer = pipeline("sentiment-analysis", truncation=True, max_length = max, device=device)
    results = sentiment_analyzer(df['TXT'].tolist()) #512 with shorter model did not work
    df['sentiment'] = [result['label'] for result in results]
    df['confidence'] = [result['score'] for result in results]
    return df




