import seaborn as sns
import matplotlib.pyplot as plt

def Visualize(data,hueKey=None):
    if(data is None):
        return

    g = sns.pairplot(data, hue=hueKey, palette='seismic', size=1.2)
    g.set(xticklabels=[])

    colormap = plt.cm

    plt.figure(figsize=(14,12))
    plt.title('Correlation of features', y = 1.05, size = 15)
    sns.heatmap(data.astype(float).corr(), linewidths=0.1, vmax=1,square=True,linecolor='white', annot=True)

    plt.show()
