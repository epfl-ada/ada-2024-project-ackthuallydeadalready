# Factors influencing voter behaviour in Wikipedia Request for Adminship Elections

## Abstract

Wikipedia administrators are a special class of editors that possess extra capabilities such as blocking user accounts, editing protected pages, etc. They have a responsibility to use the tools sparingly and in the right cases, therefore requiring a high level of trust from the community. This is important because the request for adminship (RfA) is a social consensus building process, where editors aim to make sure of the quality of the admin. The corpus we have access to is the public decision making process of each voter, including their comments and their vote (positive/negative/neutral), often including references to other voters and even reactions of other voters. 

Our project aims to harness sentiment analysis on the comments relating to an election to help identify key factors in an election. We wish to unveil group behaviors that appear in reaction to social representation of users, admins and candidates. 


## Research Questions
- How does the positivity/negativity dimension of the public sentiment about a user change over time during an election?
- How does the positivity/negativity dimension of the public sentiment about returning candidates change between elections?
- What are the consequences for candidates of having a bias for writing negative comments?
- What specific questions in the RFA voting form sway people?
- What are the most importance topics for most users?
- What are the important criteria for the voters?

## Proposed Additional Dataset

We would like to cross-reference the original data set with the questions and answers by candidates, this would provide context for comments on which we would like to perform a sentiment analysis. Moreover, this would allows us to study the recurrent themes and subject that leads to the general representation of candidates in voters opinion. However it would require us to parse the webpages in order to acquire this data as it is not already pre aggregated.

## Methods

1. Auxiliary Data Collection
    - Scrape RFA prompt questions using beautiful soup package to join with dataset
2. Preprocessing:
    - Data cleaning in preparation for tokenizing. Would mostly involve removing stop-words and punctuation
    - Grouping the data by election and year in preparation of sentiment analysis
    - Remove the "no comment" entries
3. Exploratory Data Analysis: the group will explore the dataset to check for trends period specific and time varying trends relating to elections. Main tools will include charts such as word clouds (token frequency and polarity), bar charts (Yes/No), time series (Participants per election across time).
4. Sentiment analysis:  
    - Vader (Lexicon based with predefined rules)
    - BERT (Bidirectional Encoder)
    - Models from huggingface
    - library : sentence transformers sentiment classification
5. Feature Engineering: we will process the data to create new features to capture specific dynamics, i.e. negative commentstotal polar  comments to capture negative sentiment as a ratio of total polar comments. The goal will be to create relevant features for prediction models. 
6. Prediction Models: With our semantic analysis, we will test prediction models for voting patterns given the comment history. We will try at least three models (below) for linear and non-linear relationships between features and use K-fold CV to tune hyperparameters.
    - Ensemble Methods: XGBoost (non-linear non-linear, Black Box)
    - CART Decision Tree (linear & non-linear relationships, Medium interpretability) 
    - Logit Model (Linear relationship, Maximum interpretability) 

## Timeline



## Organization



## Questions for TAs

