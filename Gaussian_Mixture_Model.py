import csv
import json
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import normalize
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import AgglomerativeClustering

def readTfIdfData():
    data = pd.read_csv('tfIdf.csv')[:800]
    data.head()
    print(len(data))
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
    return lables_true[:800]



def evaluate(lables_true, lables_pred):
    results = []
    results.append(metrics.adjusted_rand_score(lables_true, lables_pred))
    results.append(metrics.homogeneity_score(lables_true, lables_pred))
    results.append(metrics.adjusted_mutual_info_score(lables_true, lables_pred))
    return results


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

######################################3
word_to_vec_data = readWordToVecData()
lable_true = readLable()
tf_data = readTfIdfData()

##################################################### word to vec
# adjusted_rand_score = []
# homogeneity_score = []
# adjusted_mutual_info_score = []
# for i in range(2, 100):
#     print("########### ", i, " ###########")
#     gmm= GMM(n_components=i).fit(word_to_vec_data)
#     labels = gmm.predict(word_to_vec_data)
#     results = evaluate(lable_true,labels)
#     adjusted_rand_score.append(results[0])
#     homogeneity_score.append(results[1])
#     adjusted_mutual_info_score.append(results[2])
#
#
#
# gmm = GMM(n_components=45).fit(word_to_vec_data)
# labels = gmm.predict(word_to_vec_data)
# print("################# With wordToVec #################")
# results = evaluate(lable_true, labels)
# print("adjusted rand score: ", results[0])
# print("homogeneity score: ", results[1])
# print("adjusted mutual info score: ", results[2])
# saveLable(labels, "./GaussianMixture/wordToVec_cluster.csv")
#
#
# plt.plot(range(2, 100), adjusted_rand_score)
# plt.xticks(range(2, 100))
# plt.xlabel("Number of Clusters")
# plt.ylabel("Adjusted Rand Score")
# plt.show()
#
# plt.plot(range(2, 100), homogeneity_score)
# plt.xticks(range(2, 100))
# plt.xlabel("Number of Clusters")
# plt.ylabel("Homogeneity Score")
# plt.show()
#
# plt.plot(range(2, 100), adjusted_mutual_info_score)
# plt.xticks(range(2, 100))
# plt.xlabel("Number of Clusters")
# plt.ylabel("Adjusted Mutual Info Score")
# plt.show()

##################################################### tf idf

adjusted_rand_score = []
homogeneity_score = []
adjusted_mutual_info_score = []
for i in range(2, 100):
    print("########### ", i, " ###########")
    gmm= GMM(n_components=i).fit(tf_data)
    labels = gmm.predict(tf_data)
    results = evaluate(lable_true,labels)
    adjusted_rand_score.append(results[0])
    homogeneity_score.append(results[1])
    adjusted_mutual_info_score.append(results[2])



gmm = GMM(n_components=45).fit(word_to_vec_data)
labels = gmm.predict(word_to_vec_data)
print("################# With tf-idf #################")
results = evaluate(lable_true, labels)
print("adjusted rand score: ", results[0])
print("homogeneity score: ", results[1])
print("adjusted mutual info score: ", results[2])
saveLable(labels, "./GaussianMixture/tf-idf_cluster.csv")


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
