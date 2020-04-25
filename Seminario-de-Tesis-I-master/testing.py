import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import model_estimator

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

dataset = pd.read_csv("Data/Tx_0x04.csv")

def splitData(dataset, validation_size):
    vector = dataset.values
    X = vector[:, 0:dataset.shape[1] - 1] # Features
    Y = vector[:, dataset.shape[1] - 1] # Target

    return train_test_split(X, Y, test_size=validation_size)

X_train, X_test, y_train, y_test = splitData(dataset, 0.20)

model_DT = { 
    'DecisionTree': DecisionTreeClassifier()
}

# Parametros de los modelos para el Test 
# DT: 2*2*3*2*4 = 96
params_DT = {
    'DecisionTree': { 
        'class_weight': ['balanced', None],
        'criterion': ['entropy', 'gini'],
        'max_features': ['sqrt', 'log2', None],
        'min_samples_split': [2, 4, 6, 8],
        'splitter': ['best', 'random']
    }
}

model_SVC = { 
    'SVC': SVC()
}

# Parametros de los modelos para el Test 
# SVC: 4*4*4*2*2*2 = 512
params_SVC = {
    'SVC': {
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'], 
        'C': [0.5, 1, 1.5, 3], 
        'degree': [2, 3, 4, 5],
        'probability': [True, False],
        'shrinking': [True, False],
        'decision_function_shape': ['ovo', 'ovr'],
        #'max_iter': [300]
    }
}

model_kNN = {
    'kNN': KNeighborsClassifier()
}

# Parametros de los modelos para el Test
# KNN: 10*2*3*5*9=2700
params_kNN = {
    'kNN': { 
        'n_neighbors': list(range(1,11)),
        'weights': ['uniform', 'distance'], 
        'algorithm': ['ball_tree','kd_tree','brute'], 
        'p': [1, 2, 3, 4, 5],
        'leaf_size': list(range(10, 51, 5))
    }
}
'''
helperExh_DT = model_estimator.EstimatorSelection(model_DT, params_DT)
helperExh_DT.fitModel('Exh', X_train, y_train, scoring='accuracy', n_jobs=8)
print("Tiempo de ejecución (seg): "+str(helperExh_DT.timeModel['DecisionTree']))
print(helperExh_DT.scoreModel().sort_values(['.Accuracy'], ascending=False).head(10))
'''

helperExh_SVC = model_estimator.EstimatorSelection(model_SVC, params_SVC)
helperExh_SVC.fitModel('Exh', X_train, y_train, scoring='accuracy', n_jobs=4, population_size=20, generations_number=10)
print("Tiempo de ejecución (seg): "+str(helperExh_SVC.timeModel['SVC']))
print(helperExh_SVC.scoreModel().sort_values(['.Accuracy'], ascending=False).head(10))
'''
helperExh_kNN = model_estimator.EstimatorSelection(model_kNN, params_kNN)
helperExh_kNN.fitModel('Exh', X_train, y_train, scoring='accuracy', n_jobs=4, population_size=20, generations_number=10)
print("Tiempo de ejecución (seg): "+str(helperExh_kNN.timeModel['kNN']))
print(helperExh_kNN.scoreModel().sort_values(['.Accuracy'], ascending=False).head(10))

helperRnd_DT = model_estimator.EstimatorSelection(model_DT, params_DT)
helperRnd_DT.fitModel('Rdn', X_train, y_train, scoring='accuracy', n_jobs=8)
print("Tiempo de ejecución (seg): "+str(helperRnd_DT.timeModel['DecisionTree']))
print(helperRnd_DT.scoreModel().sort_values(['.Accuracy'], ascending=False).head(10))

helperRdn_SVC = model_estimator.EstimatorSelection(model_SVC, params_SVC)
helperRdn_SVC.fitModel('Rdn', X_train, y_train, scoring='accuracy', n_jobs=4, population_size=20, generations_number=10)
print("Tiempo de ejecución (seg): "+str(helperRdn_SVC.timeModel['SVC']))
print(helperRdn_SVC.scoreModel().sort_values(['.Accuracy'], ascending=False).head(10))

helperRdn_kNN = model_estimator.EstimatorSelection(model_kNN, params_kNN)
helperRdn_kNN.fitModel('Rdn', X_train, y_train, scoring='accuracy', n_jobs=4, population_size=20, generations_number=10)
print("Tiempo de ejecución (seg): "+str(helperRdn_kNN.timeModel['kNN']))
print(helperRdn_kNN.scoreModel().sort_values(['.Accuracy'], ascending=False).head(10))

helperEvol_DT = model_estimator.EstimatorSelection(model_DT, params_DT)
helperEvol_DT.fitModel('Evol', X_train, y_train, scoring='accuracy', n_jobs=8, population_size=10, generations_number=9)
print("Tiempo de ejecución (seg): "+str(helperEvol_DT.timeModel['DecisionTree']))
print(helperEvol_DT.scoreModel().sort_values(['.Accuracy'], ascending=False).head(10))

helperEvol_SVM = model_estimator.EstimatorSelection(model_SVC, params_SVC)
helperEvol_SVM.fitModel('Evol', X_train, y_train, scoring='accuracy', n_jobs=4, population_size=20, generations_number=10)
print("Tiempo de ejecución (seg): "+str(helperEvol_SVM.timeModel['SVC']))
print(helperEvol_SVM.scoreModel().sort_values(['.Accuracy'], ascending=False).head(10))

helperEvol_kNN = model_estimator.EstimatorSelection(model_kNN, params_kNN)
helperEvol_kNN.fitModel('Evol', X_train, y_train, scoring='accuracy', n_jobs=4, population_size=20, generations_number=10)
print("Tiempo de ejecución (seg): "+str(helperEvol_kNN.timeModel['kNN']))
print(helperEvol_kNN.scoreModel().sort_values(['.Accuracy'], ascending=False).head(10))
'''
