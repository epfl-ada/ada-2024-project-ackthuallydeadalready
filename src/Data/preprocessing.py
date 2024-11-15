import sys
import numpy as np
import pandas as pd
import gzip
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from bs4 import BeautifulSoup

# Parsing and importing datas
def parse_vote(vote_str):
    vote_dict = {}
    lines = vote_str.split('\n')
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            vote_dict[key.strip()] = value.strip()
    return vote_dict

def clean_txt_column(df):
    # Remove ''' - and -- characters
    df = df.str.replace(r"['-]+", "", regex=True)
    return df

def import_RFA():
    data = []
    with gzip.open('Data/wiki-RfA.txt.gz', 'rt') as file:
        content = file.read()

        # Split votes using blank lines
        votes = content.strip().split('\n\n')

        # Process each vote
        for vote in votes:
            vote_data = parse_vote(vote)
            data.append(vote_data)

    df = pd.DataFrame(data)

    df['VOT'] = pd.to_numeric(df['VOT'],errors='coerce')
    df['RES'] = pd.to_numeric(df['RES'],errors='coerce')
    #df['YEA'] = pd.to_numeric(df['YEA'],errors='coerce')
    df['TXT']=clean_txt_column(df['TXT'])

    return df

def scrape():
    #we can easily get the names of successful candidates here
    #we use names so we can also easily crosscheck with the wiki adminship data
    url = "https://en.wikipedia.org/w/api.php?action=query&list=categorymembers&cmtitle=Category:Successful_requests_for_adminship&cmlimit=20"
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')
    all_names = {}
    for s in soup.find_all('span', {'class' : 's2'}):
        text = s.getText()
        if(text[1:19] != "Wikipedia:Requests"):
            continue
        print(s.getText()[34:-1])
        all_names[text] = {}

    #wikipedia api has some limitation rate but we do not have that much data to scrape, it should be good
    all_names = {"(aeropagitica)": {}}
    for name in all_names:
        url = "https://en.wikipedia.org/w/api.php?action=parse&prop=text&page=Wikipedia:Requests_for_adminship/" + name + "&format=json"
        r = requests.get(url)
        soup = BeautifulSoup(r.content, 'html.parser')
        first_q = soup.find('b', string="1.").next_sibling
        question_one = ""
        while(first_q.name != "dl"):
            question_one += first_q.getText()
            first_q = first_q.next_sibling
        answer_one = first_q.getText()
        all_names[name][question_one] = answer_one

    return all_names, answer_one

