import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


############################ Histograma + BoxPlot
def faz_grafico(df,nome):
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(20,6))
    fig.suptitle(pala(nome))
    sns.histplot(data=df, x=nome, hue="diagnosis",alpha=.5,ax=ax1,palette=['red','blue'])
    sns.boxplot(y="diagnosis", x=nome, hue="diagnosis", data=df,ax=ax2,palette=['red','blue'])

def pala(nome):
    a=nome.split('_')
    c = []
    for b in a:
        c.append(b.title())
    a=' '.join(c)
    return a


######################### Matriz de Confusão
def matrix(df):
    f = plt.figure(figsize=(8, 6))
    plt.matshow(df.iloc[:,1:].corr(), fignum=f.number)
    plt.xticks(range(df.select_dtypes(['number']).shape[1]-1), df.select_dtypes(['number']).columns[1:], fontsize=14, rotation=90)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]-1), df.select_dtypes(['number']).columns[1:], fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Matriz de correlação', fontsize=16);