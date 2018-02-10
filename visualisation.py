import seaborn as sns
import matplotlib.pyplot as plt

def Visualize(data,hueKey=None):
    if(data is None):
        return

    g = sns.pairplot(data,hue=hueKey, palette='seismic', size=1.2)
    g.set(xticklabels=[])
    plt.show()