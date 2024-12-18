#import src.utils.preprocessing
from keybert import KeyBERT

def example_input():
    # Sample questions and answers
    qa_pairs = [
        {
            'question': 'Usually Admins Ive seen around involve themselves in keeping watch over at least one contentious topic, whether it be Politics of the United States, or Israeli–Palestinian conflict or Grand Theft Auto, or whatever else. Do you plan to focus your prospective Sysop powers onto the subject of any of these, or any other contentious topic?',
            'answer': 'Ive generally stayed away from admining contentious topics as I have spent so much time sitting on the committee, implementing and updating them. As such, Id rather not risk WP:INVOLVEment in the areas - plus, in addition, the topics dont particularly interest me. Im afraid I cant say I expect to spend my time at any contentious topics.'
        },
        {
            'question': 'How much time do you have to be doing administrator work? It is very demanding.',
            'answer': 'Administrative tasks take up about 40% of my time and are very demanding.'
        },
        {
            'question': 'what do you do when someone doesnt follow the rules',
            'answer': 'i like to show collaborative help before banning someone'
        },
        {
            'question': 'What strategies do you use to stay organized?',
            'answer': 'I use project management tools and daily prioritization to stay organized.'
        }
    ]

    # Combine question and answer for context
    return [f"{qa['question']} {qa['answer']}" for qa in qa_pairs],qa_pairs

def extract_topic(combined_texts,qa_pairs):
    # Initialize the KeyBERT model
    kw_model = KeyBERT()

    # Extract concise topics
    topics = []
    for text in combined_texts:
        #print(text)
        keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=1)
        topics.append(keywords[0][0] if keywords else "No topic found")

    #Display each Q&A pair with its main topic
    for i, (qa, topic) in enumerate(zip(qa_pairs, topics)):
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print(f"Main Topic: {topic}\n")


#make one that takes the question and answer pair dataframe and goes through, returning the dataframe with an additional column named topic
def extract_topic_from_df(df, model=KeyBERT()):
    def extract_topic(text):
        keywords = model.extract_keywords(text, keyphrase_ngram_range=(1, 3),
                                          stop_words='english', top_n=1)
        return keywords[0][0] if keywords else "No topic found"
    
    df['topic'] = (df['Question'] + df['Answer']).apply(extract_topic)
    return df






