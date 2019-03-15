import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from math import sqrt


#ユークリッド距離
def euclidean_distance(target,x):
    print(x)
    target = target.values[0]
    x = x.values
    score = 1.0/(1.0 + np.linalg.norm(target - x))
    return score

#cos類似度
def cos_similar(target,x):
    target = target.values[0]
    x = x.values
    score = np.dot(target,x)/(np.linalg.norm(target)*np.linalg.norm(x))
    return score

if __name__ =='__main__':
    df = pd.read_csv("../data/eiga_user_data_5000.csv")
    df.index.name = "id"
    df = df.reset_index()
    target = df[df["id"] == 100]
    target = target.fillna(0)
    target = target.drop(['user_id','id'],axis=1)
    df = df.fillna(0)
    df = df.drop(['user_id','id'],axis=1)
    df["euclidean_score"] = df.apply(lambda x:euclidean_distance(target,x),axis=1)
    print(df["euclidean_score"])
    df = df.drop('euclidean_score',axis=1)
    df["cos_score"] = df.apply(lambda x:cos_similar(target,x),axis=1)
    print(df["cos_score"])
