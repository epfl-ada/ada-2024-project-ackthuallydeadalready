import requests
from bs4 import BeautifulSoup
import csv

def scrape():
    #we can easily get the names of successful candidates here
    #we use names so we can also easily crosscheck with the wiki adminship data
    url = "https://en.wikipedia.org/w/api.php?action=query&list=categorymembers&cmtitle=Category:Successful_requests_for_adminship&cmlimit=10"
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')
    all_names = {}
    for s in soup.find_all('span', {'class' : 's2'}):
        text = s.getText()
        if(text[1:19] != "Wikipedia:Requests"):
            continue
        name = text[34:-1].replace(" ", "_")
        all_names[name] = {}

    #wikipedia api has some limitation rate but we do not have that much data to scrape, it should be good
    for name in all_names:
        url = "https://en.wikipedia.org/w/api.php?action=parse&prop=text&page=Wikipedia:Requests_for_adminship/" + name + "&format=json"
        r = requests.get(url)
        soup = BeautifulSoup(r.content, 'html.parser')
        first_q = soup.find('b', string="1.")
        if(first_q == None):
            continue
        first_q = first_q.next_sibling
        question_one = ""
        while(first_q != None and first_q.name != "dl"):
            question_one += first_q.getText()
            first_q = first_q.next_sibling
        if first_q == None:
            answer_one = soup.find('b', string="1.").parent.find_next_sibling('dl')
        else:
            answer_one = first_q.getText()
        all_names[name][question_one] = answer_one

    return all_names

def scraping_res():
    all_names = scrape()

    for key, value in all_names.items():
        print(key, ':', value)
    
    return None

def get_csv():
    all_names = scrape()

    with open("questions_answers.csv", "w", newline="") as f:
        w = csv.writer(f, delimiter='$')
        w.writerow(['User', 'Questions', 'Answers'])
        for key, value in all_names.items():
            for key2, value2 in value.items():
                w.writerow([key, key2, value2])
