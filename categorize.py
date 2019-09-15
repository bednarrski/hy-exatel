from os import listdir
from os.path import isfile, join
from guesslang import Guess
import warnings
warnings.filterwarnings("ignore")
import pickle
import string
from langdetect import detect
from gensim.models import LdaModel
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim

stopwords = set(stopwords.words('english'))
punctuation = set(string.punctuation) 
lemmatize = WordNetLemmatizer()

loading = LdaModel.load('topic.model')
dictionary = gensim.corpora.Dictionary.load('dictionary.dict')

def cleaning(article):
    one = " ".join([i for i in article.lower().split() if i not in stopwords])
    one = [i for i in str(one) if i not in punctuation]
    two = "".join(one)
    three = " ".join(lemmatize.lemmatize(i) for i in two.split())
    return three


def cleaning_fail(article):
    one = " ".join([i for i in article.lower().split() if i not in stopwords])
    three = " ".join(lemmatize.lemmatize(i) for i in one.split())
    return three

def pre_new(doc):
    try:
        one = cleaning(doc).split()
    except:
        one = cleaning_fail(doc).split()
    two = dictionary.doc2bow(one)
    return two

def topic_string(topics_found, topics_names_dict):
    results = []
    for item in topics_found:
        key = topics_names_dict[item[0]]
        val = item[1]
        if val > 0.1:
            results.append((key, val))
    
    topics = str(results)
    return topics

if __name__ == "__main__":
    
    processed_count = 0
    limit = 20000

    mypath = "data/hackyeah_data_80"
    folders = [join(mypath, f) for f in listdir(mypath) if not isfile(join(mypath, f))]
    
    count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
    
    loading = LdaModel.load('topic.model')
    
    topics_names_dict = {0: "server logs",
                        1: "online gaming",
                        2: "server logs",
                        3: "online gaming",
                        4: "server logs",
                        5: "server logs",
                        6: "online gaming",
                        7: "torrent info",
                        8: "online gaming",
                        9: "torrent info",
                        10: "torrent info",
                        11: "torrent info",
                        12: "online gaming",
                        13: "torrent info",
                        14: "torrent info",
                        15: "trading",
                        16: "server logs",
                        17: "meaningful content",
                        18: "meaningful content",
                        19: "meaningful content"
                    }

    for folder in folders:
        files = [f for f in listdir(folder) if isfile(join(folder, f))]

        for f in files:
            file = join(folder, f)
            description = ""
            not_recognized = False
            
            with open(file, 'r') as file_reader:
                data = file_reader.read()
            data = '\n'.join([x for x in data.split("\n") if x.strip()!=''])
                
            # Stage 1 - recognize code vs human readable file.
            
            coding_langs = Guess().scores(data)

            k, v = sorted(coding_langs.items(), key=lambda p:p[1])[-1]
            if k == "Markdown" or v < 0.9:
                not_recognized = True
            else:
                description += "Code in " + k + "."
                
            if not_recognized:
                # Stage 2 - Check weird interpunction.
                punctuation_coding = count(data, string.punctuation.replace("!", "").replace(".", "").replace("?", "").replace(",", ""))
                if punctuation_coding/len(data) > 0.06:
                    description += "Not recognized code or console output."
                else:
                    # Stage 3 - Recognize language.
                    try:
                        lang = detect(data)
                    except:
                        lang = 'not recognized'
                    description += "Human readable file, language: " + lang + "."
                    
                    if lang == 'en':
                        # Stage 4 - For english find topics with LDA.
                        topics_found = loading[(pre_new(data))]
                        description += "Topics found: " + topic_string(topics_found, topics_names_dict) + "."
                        

            #print(data)
            print(f+"\t"+description)
            #print("----------------------------")
            processed_count += 1
            
            if processed_count > limit:
                break
        if processed_count > limit:
            break
