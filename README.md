# Factors influencing voter behaviour in Wikipedia Request for Adminship Elections

## Abstract

Wikipedia administrator

The Wikipedia environment has many types of user accounts that have access to different functionalities to fulfill different roles. One of these roles is administrators, which have high and broad technical privileges such as deleting and protecting pages, blocking users and editing protected pages. The administrator status is obtained at the end of a process that starts with the person or another user submitting a Request For Adminship (RFA). The entire community then has about a week to cast a positive, negative or neutral vote accompanied by comments which fuel discussions around the candidate. Finally, the final decision to promote or not is made by users called bureaucrats. In the age of social media algorithms, online anonymity and insidious internet communication shenanigans, the different factors that can impact the discourse and voting behaviour of large masses of individuals over short periods of time in the context of limited information about candidates become important to investigate.

## Abstract Proposed Change

Wikipedia administrators (admins) possess the right to maintain the database and participate in Request for Adminship (RFA) elections. Candidates for adminstrators can stand for election, sometimes with an admin co-nominator. In the one week election, admins can have a positve/negative/neutral vote. The final decision for approval is made by the bureaucrats (user-type) of Wikipedia. 

Our project will use sentiment analysis on the comments relating to an election and check if the sentiment of an election can help identify important factors in an election. We will focus on the voting admin's responses to a candidacy. The corpus is similar to a chatroom with standalone comments and replies to previous comments. Admins refer to the responses to questions in the candidate application to justify their opinion. To get a fuller picture, we will scrape the questions and responses by candidates and join the data. In the end, we want 

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

- Remove the "no comment" entries
- scrape questions https://realpython.com/beautiful-soup-web-scraper-python/
- Sentiment Analysis:
    - Vader (Lexicon based with predefined rules)
    - BERT (Bidirectional Encoder)
    - Models from huggingface 
- library : sentence transformers sentiment classification

## Timeline



## Organization



## Questions for TAs

