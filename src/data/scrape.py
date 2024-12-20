import requests
from bs4 import BeautifulSoup
import csv
import re

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



def scrape_by_name_all_qs_all_attempts(names,df):
    all_names = {}
    for name in names:
        all_names[name] = {}          
    for name in all_names:
        attempts=len(df[df['TGT']==name].drop_duplicates(subset=['TGT','YEA','RES']).value_counts())
        for attempt in range(1,attempts+1):
            all_names[name][f"Attempt {attempt}"]={}
            if (attempt==1):
                url = ("https://en.wikipedia.org/w/api.php?action=parse&prop=text&page=Wikipedia:Requests_for_adminship/" + name +
                                                                                                                    "&format=json")
            else:
                url = ("https://en.wikipedia.org/w/api.php?action=parse&prop=text&page=Wikipedia:Requests_for_adminship/" + name + " " 
                                                                                                    + str(attempt) + " " 
                                                                                                    + "&format=json")
            r = requests.get(url)
            soup = BeautifulSoup(r.content, 'html.parser')
            i = 1
            while True:
                question=soup.find('b', string=(lambda text: text == f"{i}." or text == f"{i}"))
                if(question == None):
                    break
                question = question.next_sibling
                question_i = ""
                while question and question.name not in ["dl", "dd", "b"]:
                    if question.string:
                        question_i += question.get_text(strip=True)
                    question = question.next_sibling
                question_i = question_i.replace("\\n","").replace("\\",'')
                answer_i = None
                parent = soup.find('b', string=(lambda text: text == f"{i}." or text == f"{i}"))
                if parent:
                    answer_i = parent.find_next('dl')           
                if answer_i and question_i:
                    all_names[name][f"Attempt {attempt}"][f"Question {i}"] = {
                        "Question": question_i.strip(),
                        "Answer": re.sub(r'A[:.]','', answer_i.get_text(strip=True))
                    }
                else:
                    all_names[name][f"Attempt {attempt}"][f"Question {i}"] = {
                        "Question": question_i.strip(),
                        "Answer": "No answer found"
                    }
                    print(i)
                    print(name)
                    print(f"answer_i type: {type(answer_i)}")  # Check type
                    print(f"answer_i value: {answer_i}")  # Check value        
                            
                i += 1
    scraper.dict_to_csv(all_names,'res/data/all_questions_and_answers.csv')
    return all_names



def dict_to_csv(data_dict, filename):
    with open(filename, "w", newline="") as f:
        w = csv.writer(f, delimiter='$')
        w.writerow(['User','Attempt', 'Question', 'Answer'])

        for user, attempts in data_dict.items():
            for attempt, questions in attempts.items():
                for question, answer_dict in questions.items():
                    question_text = answer_dict.get('Question', '')
                    answer_text = answer_dict.get('Answer', '')
                    w.writerow([user, int(attempt[-2:]),question_text, answer_text]) #assumption, no tripl digit number of attempts
