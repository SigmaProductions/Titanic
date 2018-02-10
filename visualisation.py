import seaborn as sns
import matplotlib.pyplot as plt

def Visualize(data):
    if(data==None):
        return

    g = sns.pairplot(data[[u'Survived', u'Pclass', u'Sex', u'Age', u'Parch', u'Fare', u'Embarked',
                            ]], hue='Survived', palette='seismic', size=1.2, diag_kind='kde', diag_kws=dict(shade=True),
                     plot_kws=dict(s=10))
    g.set(xticklabels=[])
    plt.show()