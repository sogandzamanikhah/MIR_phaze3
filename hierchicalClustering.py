import csv
import json
import pickle
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering

def readTfIdfData():
    data = pd.read_csv('tfIdf.csv')
    data.head()
    data_scaled = normalize(data)
    data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
    return data_scaled

def readWordToVecData():
    data = pd.read_csv('wordToVec.csv')
    data.head()
    data_scaled = normalize(data)
    data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
    return data_scaled

def readLable():
    with open("tagList.txt", "rb") as data:
        lables_true = pickle.load(data)
    return lables_true


def clusterWithTfIdf(data):
    cluster = AgglomerativeClustering(n_clusters=49, affinity='euclidean', linkage='ward')
    lables_pred=cluster.fit_predict(data)
    return lables_pred

def clusterWithWordToVec(data):
    cluster = AgglomerativeClustering(n_clusters=49, affinity='euclidean', linkage='ward')
    lables_pred=cluster.fit_predict(data)
    return lables_pred


def evaluateTfIdf(lables_true,lables_pred):
    print("withTfIdf")
    print("metrics.rand_score",metrics.rand_score(lables_true, lables_pred))
    print("metrics.homogeneity_score",metrics.homogeneity_score(lables_true, lables_pred))
    print("metrics.adjusted_mutual_info_score",metrics.adjusted_mutual_info_score(lables_true, lables_pred))


def evaluateWordToVec(lables_true,lables_pred):
    print("withWordToVec")
    print("metrics.rand_score",metrics.rand_score(lables_true, lables_pred))
    print("metrics.homogeneity_score",metrics.homogeneity_score(lables_true, lables_pred))
    print("metrics.adjusted_mutual_info_score",metrics.adjusted_mutual_info_score(lables_true, lables_pred))


def saveLable(lable_pred,file_name):
    f = open('hamshahri.json', )
    data = json.load(f)
    dataset=[]
    for i in data:
        dataset.append([i["link"][:43]])
    f.close()
    for i in range(0,len(lable_pred)):
        dataset[i].append(lable_pred[i])
    f = open(file_name, 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(["link","lable"])
    for i in dataset:
        writer.writerow(i)


tf_data = readTfIdfData()
lable_true = readLable()
lable_pred = clusterWithTfIdf(tf_data)
saveLable(lable_pred,"./Hierarchical/tfIdf_cluster.csv")
evaluateTfIdf(lable_true,lable_pred)

word_to_vec_data = readWordToVecData()
lable_pred_word_to_vec = clusterWithWordToVec(word_to_vec_data)
saveLable(lable_pred_word_to_vec,"./Hierarchical/wordToVec_cluster.csv")
evaluateWordToVec(lable_true,lable_pred_word_to_vec)












