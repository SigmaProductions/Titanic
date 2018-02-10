import tensorflow as tf
import pandas as ps
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import visualisation as vis
import random
import classifier as clas
def processFeatures(train):


	#calculate mean age
    ageAvr= train["Age"].mean()
    ageStd= train["Age"].std()
    ageNullCount = train['Age'].isnull().sum()
    rndNullList = np.random.randint(ageAvr - ageStd, ageAvr + ageStd, size=ageNullCount)
    
    train['Age'][np.isnan(train['Age'])]= rndNullList
    train['Age'] = train['Age'].astype(int)


    train['Sex'] = train['Sex'].map({'female': 0, 'male': 1}).astype(int)
    train['Embarked'] = train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2 , None:random.randint(-1,2)}).astype(int)
    train['Cabin'] = train['Cabin'].apply(lambda x: 0 if type(x) == float else 1)
    dropElements = ['PassengerId', 'Name', 'Ticket']
    train =  train.drop(dropElements, axis=1)
    print(train.head(10))
    return train

if __name__ == "__main__":
    train = ps.read_csv("data/train.csv")
    train = processFeatures(train)
    #clas.learn(train)
    vis.Visualize(train, "Survived")