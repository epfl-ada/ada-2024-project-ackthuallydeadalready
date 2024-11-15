import requests
from bs4 import BeautifulSoup

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

def scraping_res():
    all_names, answer_one = scrape()

    for key, value in all_names["(aeropagitica)"].items():
        print(key)
        print(value)
    
    return None