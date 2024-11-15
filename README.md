# Abstract 

Wikipedia administrators are a special class of editors that possess extra capabilities such as blocking user accounts, editing protected pages, etc. They have a responsibility to use the tools sparingly and in the right cases, therefore requiring a high level of trust from the community. This is important because the request for adminship (RfA) is a social consensus building process, where editors aim to make sure of the quality of the admin. Bureaucrats evaluate the general consensus to make a decision, the votes being only indicative.
Our project aims to harness sentiment analysis on the comments to help extract and understand key topics that may strongly influence the outcome of an election and how users, admins and candidates interact around these contentious topics. 
This can help us understand consensus building process social dynamics and also in designing more effective consensus systems which address simple voting limitations.
Proposed Additional Dataset
We would like to cross-reference the original data set with the questions and answers by candidates, this would provide context for comments on which we would like to perform a sentiment analysis. Moreover, this would allow us to study the recurrent themes and subjects that lead to the general representation of candidates in voters opinion. It will require us to parse the webpages in order to acquire this data as it is not already pre aggregated.
Wikipedia has its own API for web scraping so it is easy to gather the data. 

# Research Questions

Which topics (e.g., policy stance, previous contributions, communication skills) provoke the most sentiment-laden comments? 
For this we will use the additional dataset of questions to the candidate.
We first find all the references to the questions in the comments. We then conduct a sentiment analysis of those comments and extract the ones with the strongest reactions(positive or negative) by using the polarity score or label probability. Next, out of those comments we extract the corresponding questions and their main topic. 
Which topics have swayed people’s opinion ?. We will  find which voters have changed their minds (this is easy to do as the history is in the comments). For all of these voters, we will aim to extract what topic made them change their opinion by analyzing the key topics when they changed their votes. Do these topics correspond with the ones found using the questions and sentiment analysis in point 1)?
Do certain questions correlate with higher engagement?
Out of the topics in point 1), we find in each RFA whether they are overrepresented in the comments. This could help us find whether these topics are highly contentious.  
Are the most important topics more often associated with either positive or negative sentiments?or both?
We will use topics found in point 1) and then find out if they are linked to strong sentiments in various RFAs. 
Which topics appear most frequently in the entire corpus?
We then can find which topics are referenced the most. We will evaluate the whole dataset, divide it by each election and see whether there is a trend every year. Maybe some topics are important in the beginning of admin selection process and it began to die down over the years. 
Do certain voters often talk about certain topics only?
We will follow some highly engaged users and see if they often mention some of these contentious topics, or have a preference to mention some of these topics very often. In this case, do they tend to feel more positively or negatively (or both) towards them?This could help us to characterize voter habits.


# Methods

Auxiliary Data Collection
Scrape RFA prompt questions using beautiful soup package to join with dataset
Preprocessing:
Data cleaning in preparation for tokenizing. Would mostly involve removing stop-words and punctuation
Grouping the data by election and year in preparation of sentiment analysis
Remove the "no comment" entries
Exploratory Data Analysis: the group will explore the dataset to check for trends period specific and time varying trends relating to elections. Main tools will include charts such as word clouds (token frequency and polarity), bar charts (Yes/No), time series (Participants per election across time).
Sentiment analysis: we will use different models to classify tokens and see if different NLP models will lead to different results and if this will lead to difference in the topic analysis stage
Vader (Lexicon based with predefined rules)
BERT (Bidirectional Encoder)
Models from huggingface
library : sentence transformers sentiment classification
Topic Analysis: Using Latent Dirichlet Analysis (LDA), we can infer which topics are relevant for our analysis. This will be later used to check whether comments relating to these topics have any predictive power on future 
Feature Engineering: we will process the data to create new features to capture specific dynamics, i.e. negative commentstotal polar  comments to capture negative sentiment as a ratio of total polar comments. The goal will be to create relevant features for prediction models. 
Prediction Models: With our semantic analysis, we will test prediction models for voting patterns given the comment history. We will try at least three models (below) for linear and non-linear relationships between features and use K-fold CV to tune hyperparameters. 
Ensemble Methods: XGBoost (non-linear non-linear, Black Box)
CART Decision Tree (linear & non-linear relationships, Medium interpretability) 
Logit Model (Linear relationship, Maximum interpretability) 

# Timeline
15.11 - Submitted Milestone P2 (START)
29.11 - HW2 Due
29.11 - Task 1: Scraping and preprocessing of all raw data finished
subtasks: data-scraping, joining datasets, tokenization of comments, cleaning tokens' data (stopwords and punctuation)
06.11 - Task 2: Exploratory Data Analysis finished
subtasks: describe preliminary results and hypotheses with visual evidence and summary statistics
10.12 - Task 3: Prediction Models finished
subtasks: XGBoost, CART, Logit Model
10.12 - Task 4: Creation of Github Webpage
subtasks: template selection, graph inclusion, UX of website
14.12 - Task 5: Cleaning code
subtasks: cleaning code & ensuring execution
16.12 - Task 6: Story building Finished
subtasks: story telling, visualizing insights, chart curation
20.12 - Submission Milestone P3 (END)

# Organization
Task 1 - Aidan Maxence Reinatt
Task 2 - Mikail Alexandre Reinatt
Task 3 - Mikail Maxence Reinatt
Task 4 - Aidan Alexandre
Task 5 - Maxence Alexandre
Task 6 - Mikail Aidan

# Questions for TAs


# Jupyter Notebook file:
Results

Data scraping sample
Sentiment Analysis
Visualization computing
Plots
