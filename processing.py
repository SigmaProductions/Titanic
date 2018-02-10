import tensorflow as tf
import pandas as ps
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import visualisation as vis

def processFeatures():
    train = ps.read_csv("data/train.csv")

    #calculate mean age
    ageAvr= train["Age"].mean()
    ageStd= train["Age"].std()
    ageNullCount = train['Age'].isnull().sum()
    rndNullList = np.random.randint(ageAvr - ageStd, ageAvr + ageStd, size=ageNullCount)
    
    train['Age'][np.isnan(train['Age'])]= rndNullList
    train['Age'] = train['Age'].astype(int)

    vis.Visualize(train[["Age"]]);

if __name__ == "__main__":
    processFeatures()
