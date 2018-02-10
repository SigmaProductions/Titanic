import tensorflow as tf
import pandas as ps
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import visualisation as vis
import random

def processFeatures():
    train = ps.read_csv("data/train.csv")
    train['Sex'] = train['Sex'].map({'female': 0, 'male': 1}).astype(int)
    train['Embarked'] = train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2 , None:random.randint(-1,2)}).astype(int)
    train['Cabin'] = train['Cabin'].apply(lambda x: 0 if type(x) == float else 1)
    dropElements = ['PassengerId', 'Name', 'Ticket']
    train =  train.drop(dropElements, axis=1)
    print(train.head(10))
    #vis.Visualize(None);

if __name__ == "__main__":
    processFeatures()
