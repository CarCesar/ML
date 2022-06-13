import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve

def plot_confusion_matrix(cm, classes):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds, aspect='auto')
    plt.title('Matriz de Confusão')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predição')
    
def evalu(clf, X, y_true, nome, Dados, classes=['Benigno', 'Maligno']):
    y_pred = clf.predict(X)
    clf_matrix = confusion_matrix(y_true, y_pred)
    print('Resultados:')
    print(classification_report(y_true, y_pred, target_names=classes))
    print('ROC: {}'.format(roc_auc_score(y_true, y_pred)))
    print('Acurácia : {}'.format(accuracy_score(y_true, y_pred)))
    print('Precisão : {}'.format(average_precision_score(y_true, y_pred)))
    print('Recall : {}'.format(recall_score(y_true, y_pred)))
    print('f1 : {}'.format(f1_score(y_true, y_pred)))
    plot_confusion_matrix(clf_matrix, classes=classes)
    Dados = Dados.append({'modelo':nome,'acuracia':accuracy_score(y_true, y_pred),'recall':recall_score(y_true, y_pred)},
                 ignore_index=True)
    return Dados