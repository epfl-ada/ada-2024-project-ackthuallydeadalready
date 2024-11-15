import src.utils.plotting
from collections import Counter
from nltk.corpus import stopwords
import nltk

# Define a function to remove stop words from a sentence
def remove_stop_words(sentence):
    stop_words = set(stopwords.words('english'))
    # Split the sentence into individual words
    words = sentence.split()

    # Use a list comprehension to remove stop words
    filtered_words = [word for word in words if word not in stop_words]

    # Join the filtered words back into a sentence
    return ' '.join(filtered_words)

def stopwords_removal(df):
    # Download NLTK data
    nltk.download('stopwords')  
    stop_words = set(stopwords.words('english'))
    all_words = ' '.join(df['TXT']).lower().split()
    print(Counter([word for word in all_words if word not in stop_words]).most_common(10))

    stop_words = set(stopwords.words('english'))

    df["Comment_no_stopword"] = df['TXT'].apply(remove_stop_words)



