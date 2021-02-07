import json
import pickle
from hazm import Stemmer, Lemmatizer, Normalizer, word_tokenize,stopwords_list
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer


stopwords = stopwords_list()

def Preprocess(doc):
    stemmer = Stemmer()
    lemmatizer = Lemmatizer()
    normalizer = Normalizer()
    doc = normalizer.normalize(doc)
    tokenized = re.split(' |-',doc)
    for w in tokenized[:]:
        if w in stopwords:
            tokenized.remove(w)
    stemmed = [stemmer.stem(w) for w in tokenized]
    new_words = [word for word in stemmed if word.isalnum() ]
    lemmatized = [lemmatizer.lemmatize(w) for w in new_words]
    S=""
    for w in lemmatized:
        S=S+w+" "
    return S


def fetchTag():
    tag_set={" "}
    tag_list=[]
    f = open('hamshahri.json')
    data = json.load(f)
    for doc in data:
        tag_set.update(doc["tags"])
        tag_list = tag_list + doc["tags"]
    # print(tag_list)
    # print(len(tag_set))
    with open("tagList.txt", "wb") as output:
        pickle.dump(tag_list, output)


def fetchData():
    f = open('hamshahri.json', )
    data = json.load(f)
    dataset=[]
    for i in data:
        dataset.append(i["title"]+" "+i["summary"]+i["link"][43:])
    f.close()
    return dataset


def makeTfIdf(dataset):
    tfIdfVectorizer=TfidfVectorizer(preprocessor= Preprocess)
    tfIdf = tfIdfVectorizer.fit_transform(dataset)
    nlp_name = pd.DataFrame(tfIdf.toarray(), columns=tfIdfVectorizer.get_feature_names())
    nlp_name.to_csv(r'tfIdf.csv')

dataset=fetchData()
# print(len(dataset))
makeTfIdf(dataset)
fetchTag()


