import csv
import json
import re
import numpy as np

import gensim as gensim
from hazm import Stemmer, Lemmatizer, Normalizer, stopwords_list

stopwords = stopwords_list()
def preprocess(doc):
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
    return lemmatized

def fetchData():
    f = open('hamshahri.json', )
    data = json.load(f)
    dataset=[]
    for i in data:
        sentence=i["title"]+" "+i["summary"]+i["link"][43:]
        dataset.append(preprocess(sentence))
    f.close()
    # print(len(dataset))
    return dataset

def makeWordToVec(dataset):
    f = open("WordToVec.csv", 'w', newline='')
    writer = csv.writer(f)
    writer.writerow([" " for x in range(32)])
    model1 = gensim.models.Word2Vec(dataset, min_count = 1, size = 32)
    word_to_Vec_list = []
    vector_doc = []
    for doc in dataset:
        for word in doc:
            word_to_Vec_list.append(model1.wv[word])
        data = np.array(word_to_Vec_list)
        x=np.average(data, axis=0)
        writer.writerow(x)
        vector_doc.append(x)
        word_to_Vec_list = []
    # print(len(vector_doc))


dataset = fetchData()
makeWordToVec(dataset)



