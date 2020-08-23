import pandas as pd
import numpy as np
import time, random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier as rfr
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.svm import SVC as svm
from sklearn.utils import resample

class Classifier:
    def __init__(self, df):
        self.df = df
        self.X = None 
        self.y = None
        self.Xtrain = None
        self.ytrain = None
        self.Xtest = None
        self.ytest = None
        self.report = None
        self.best = None

    def dataset_split(self, balanced=False):
        
        if balanced:
            df_majority = self.df[self.df.IsSpam==0]
            df_minority = self.df[self.df.IsSpam==1]

            df_majority_downsampled = resample(df_majority, replace=False, n_samples=len(self.df[self.df.IsSpam==1]), random_state=123)
            df_downsampled = pd.concat([df_majority_downsampled, df_minority])
            self.df = df_downsampled
        
        le = LabelEncoder()
        self.df["IsSpam"] = le.fit_transform(self.df["IsSpam"])

        y = self.df["IsSpam"]
        X = self.df.drop("IsSpam", axis=1)
        X = X.iloc[:,1:-1] # remove unnecessary features Full-Text and Date

        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        self.X = X
        self.y = y
        self.Xtrain = Xtrain
        self.Xtest = Xtest
        self.ytrain = ytrain
        self.ytest = ytest

    def evaluate_models(self):

        models = {}
        models["Random forest classifier"] =  rfr()
        models["K nearest neighbors classifier K3u"] = knn(n_neighbors=3, weights="uniform")  
        models["K nearest neighbors classifier K3d"] = knn(n_neighbors=3, weights="distance")  
        models["K nearest neighbors classifier K5"] = knn(n_neighbors=5) 
        models["Decision tree classifier"] = dtc()
        models["SVM"] = svm()

        report = {"Model":[], "Score":[], "Elapsed Time(s)":[]}
        for model_name in models:
            start = time.time()
            model = models[model_name].fit(self.Xtrain, self.ytrain)
            report["Elapsed Time(s)"].append(time.time()-start)
            ypred = model.predict(self.Xtest)
            report["Score"].append(accuracy_score(ypred, self.ytest))
            report["Model"].append(model_name)  

        report = pd.DataFrame.from_dict(report)
        report.sort_values(by="Score", inplace=True)
        report.reset_index(inplace=True, drop=True)
        best = report[-1:]
        self.report = report
        self.best = best

