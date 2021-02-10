from sklearn.cluster import KMeans
import numpy as np
import csv
import json
import pickle
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt


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


def evaluateTfIdf(lables_true, lables_pred):
    # print("withTfIdf")
    results = []
    results.append(metrics.adjusted_rand_score(lables_true, lables_pred))
    results.append(metrics.homogeneity_score(lables_true, lables_pred))
    results.append(metrics.adjusted_mutual_info_score(lables_true, lables_pred))
    return results


def saveLable(lable_pred, file_name):
    f = open('hamshahri.json', )
    data = json.load(f)
    dataset = []
    for i in data:
        dataset.append([i["link"]])
    f.close()
    for i in range(0, len(lable_pred)):
        dataset[i].append(lable_pred[i])
    f = open(file_name, 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(["link", "lable"])
    for i in dataset:
        writer.writerow(i)

# word_to_vec_data = readWordToVecData()

# tf_data = readTfIdfData()
word_to_vec_data = readWordToVecData()
lable_true = readLable()

adjusted_rand_score = []
homogeneity_score = []
adjusted_mutual_info_score = []
for i in range(2, 100):
    print("########### ", i, " ###########")
    kmeans = KMeans(n_clusters=i).fit(word_to_vec_data)
    results = evaluateTfIdf(lable_true, kmeans.labels_)
    adjusted_rand_score.append(results[0])
    homogeneity_score.append(results[1])
    adjusted_mutual_info_score.append(results[2])


kmeans = KMeans(n_clusters=58).fit(word_to_vec_data)
print("################# With wordToVec #################")
results = evaluateTfIdf(lable_true, kmeans.labels_)
print("adjusted rand score: ", results[0])
print("homogeneity score: ", results[1])
print("adjusted mutual info score: ", results[2])
saveLable(kmeans.labels_, "./Kmeans/wordToVec_cluster.csv")


plt.plot(range(2, 100), adjusted_rand_score)
plt.xticks(range(2, 100))
plt.xlabel("Number of Clusters")
plt.ylabel("Adjusted Rand Score")
plt.show()

plt.plot(range(2, 100), homogeneity_score)
plt.xticks(range(2, 100))
plt.xlabel("Number of Clusters")
plt.ylabel("Homogeneity Score")
plt.show()

plt.plot(range(2, 100), adjusted_mutual_info_score)
plt.xticks(range(2, 100))
plt.xlabel("Number of Clusters")
plt.ylabel("Adjusted Mutual Info Score")
plt.show()

# word_to_vec_data = readWordToVecData()
