# Abstract 

<p align="justify">
Wikipedia administrators are a special class of editors that possess extra capabilities such as blocking user accounts, editing protected pages, etc. They have a responsibility to use the tools sparingly and in the right cases, therefore requiring a high level of trust from the community. This is important because the request for adminship (RfA) is a social consensus building process, where editors aim to make sure of the quality of the admin. Bureaucrats evaluate the general consensus to make a decision, the votes being only indicative.<br />
Our project aims to harness sentiment analysis on the comments to help extract and understand key topics that may strongly influence the outcome of an election and how users, admins and candidates interact around these contentious topics.  <br />
This can help us understand consensus building process social dynamics and also in designing more effective consensus systems which address simple voting limitations.
</p>

# Proposed Additional Dataset
<p align="justify">
We would like to cross-reference the original data set with the questions and answers by candidates, this would provide context for comments on which we would like to perform a sentiment analysis. Moreover, this would allow us to study the recurrent themes and subjects that lead to the general representation of candidates in voters opinion. It will require us to parse the webpages in order to acquire this data as it is not already pre aggregated.<br />
Wikipedia has its own API for web scraping so it is easy to gather the data. We will scrape it directly from the url of each successful or unsuccessful candidate. Then, we will cross check the username with the user’s election that we have in the RfA dataset. By doing this, we will get all the questions and answers for each user.
</p>

# Research Questions
 
1. **Which topics (e.g., policy stance, previous contributions, communication skills) provoke the most sentiment-laden comments?**  
For this we will use the additional dataset of questions to the candidate.
We first find all the references to the questions in the comments. We then conduct a sentiment analysis of those comments and extract the ones with the strongest reactions(positive or negative) by using the polarity score or label probability. Next, out of those comments we extract the corresponding questions and their main topic. 
2. **Which topics have swayed people’s opinion ?**  
We will  find which voters have changed their minds (this is easy to do as the history is in the comments). For all of these voters, we will aim to extract what topic made them change their opinion by using topic analysis when they changed their votes. Do these topics correspond with the ones found using the questions and sentiment analysis in point 1)?
3. **Do certain questions correlate with higher engagement?**  
Out of the topics in point 1), we find in each RFA whether they are overrepresented in the comments. This could help us find whether these topics are highly contentious.  
4. **Are the most important topics more often associated with either positive or negative sentiments?or both?**  
We will use topics found in point 1) and then find out if they are linked to strong sentiments in various RFAs. 
5. **Which topics appear most frequently in the entire corpus?**  
We then can find which topics are referenced the most. We will evaluate the whole dataset, divide it by each election and see whether there is a trend every year. Maybe some topics are important in the beginning of the admin selection process and it began to die down over the years.  
6. **Do certain voters often talk about certain topics only?**  
We will follow some highly engaged users and see if they often mention some of these contentious topics, or have a preference to mention some of these topics very often. In this case, do they tend to feel more positively or negatively (or both) towards them?This could help us to characterize voter habits.


# Methods

1. **Auxiliary Data Collection**
    - Scrape RFA prompt questions using beautiful soup package to join with dataset
    - Clean the questions and answers to a good format
2. **Preprocessing:**
    - Data cleaning in preparation for tokenizing. Would mostly involve removing stop-words and punctuation
    - Grouping the data by election and year in preparation of sentiment analysis
    - Remove the "no comment" entries
3. **Exploratory Data Analysis:**  
the group will explore the dataset to check for trends period specific and time varying trends relating to elections. Main tools will include charts such as word clouds (token frequency and polarity), bar charts (Yes/No), time series (Participants per election across time).
4. **Sentiment analysis:**  
we will use different models to classify tokens and see if different NLP models will lead to different results and if this will lead to difference in the topic analysis stage
    - Vader (Lexicon based with predefined rules)
    - BERT (Bidirectional Encoder)
    - Models from huggingface
    - library : sentence transformers sentiment classification
6. **Topic Analysis:**  
Using Latent Dirichlet Analysis (LDA), we can infer which topics are relevant for our analysis. This will be later used to check whether comments relating to these topics have any predictive power on future 
7. **Feature Engineering:**  
we will process the data to create new features to capture specific dynamics, i.e. negative commentstotal polar  comments to capture negative sentiment as a ratio of total polar comments. The goal will be to create relevant features for prediction models. 
8. **Prediction Models:**  
With our semantic analysis, we will test prediction models for voting patterns given the comment history. This prediction model will focus on sentiments and topics that have been learned. We believe for the RfA system, it should perform better than other models since bureaucrats are the ones deciding the final vote, not a simple voting system. We will try at least three models (below) for linear and non-linear relationships between features and use K-fold CV to tune hyperparameters.  
    - Ensemble Methods: XGBoost (non-linear non-linear, Black Box)
    - CART Decision Tree (linear & non-linear relationships, Medium interpretability) 
    - Logit Model (Linear relationship, Maximum interpretability) 

# Timeline (Planned)
- 15.11 - Submitted Milestone P2 (START)
- 29.11 - HW2 Due
- 29.11 - Task 1: Scraping and preprocessing of all raw data finished
    - subtasks: data-scraping, joining datasets, tokenization of comments, cleaning tokens' data (stopwords and punctuation)
- 06.11 - Task 2: Exploratory Data Analysis finished
    - subtasks: describe preliminary results and hypotheses with visual evidence and summary statistics
- 10.12 - Task 3: Prediction Models finished
    - subtasks: XGBoost, CART, Logit Model
- 10.12 - Task 4: Creation of Github Webpage
    - subtasks: template selection, graph inclusion, UX of website
- 14.12 - Task 5: Cleaning code
    - subtasks: cleaning code & ensuring execution
- 16.12 - Task 6: Story building Finished
    - subtasks: story telling, visualizing insights, chart curation
- 20.12 - Submission Milestone P3 (END)

# Organization (Planned)
- Task 1 - Aidan Maxence Reinatt
- Task 2 - Mikail Alexandre Reinatt
- Task 3 - Mikail Maxence Reinatt
- Task 4 - Aidan Alexandre
- Task 5 - Maxence Alexandre
- Task 6 - Mikail Aidan

# Accurate Tasks Performed
- Task 1: Scraping finished
    - subtasks: data-scraping, joining datasets, tokenization of comments, cleaning tokens' data (stopwords and punctuation)
- Task 2: Exploratory Data Analysis finished
    - subtasks: describe preliminary results and hypotheses with visual evidence and summary statistics
- Task 2.5: EDA on Sentiment Analysis and Topics
    - subtasks: describe preliminary results and hypotheses with visual evidence and summary statistics
- Task 3: Prediction Models finished
    - subtasks: XGBoost, Logistic Model, Linear Regression, Small Neural Network
- Task 4: Creation of Github Webpage
    - subtasks: template selection, graph inclusion, UX of website
- Task 5: Cleaning code
    - subtasks: cleaning code & ensuring execution
- Task 6: Story building Finished
    - subtasks: story telling, visualizing insights, chart curation

# Contributions
- Task 1 - Alexandre Reinatt
- Task 2 - Mikail 
- Task 2.5 - Maxence
- Task 3 - Alexandre
- Task 4 - Maxence
- Task 5 - Maxence Alexandre Mikail
- Task 6 - Mikail Alexandre Maxence

# Questions for TAs


# Jupyter Notebook file:
results.ipynb contains:
- Data scraping sample  
- Plots  
- Sentiment Analysis
- Topic Analysis
- Visualization computing  


Notes : pip requirments -> specifier pip freeze