import tensorflow as tf
import pandas as ps
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import visualisation as vis

def processFeatures():
    train = ps.read_csv("data/train.csv")
        
    vis.Visualize(None);

if __name__ == "__main__":
    processFeatures()
