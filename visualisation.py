import seaborn as sns
import matplotlib.pyplot as plt

def Visualize(data):
    if(data is None):
        return

    g = sns.pairplot(data, palette='seismic', size=1.2, diag_kind='kde', diag_kws=dict(shade=True),
                     plot_kws=dict(s=10))
    g.set(xticklabels=[])
    plt.show()