#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 8 19:23:20 2020

@author: alfredocu
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

classifiers = {
    "KNN": KNeighborsClassifier(3),
    "SVM": SVC(gamma = 2, C = 1),
    "GP": GaussianProcessClassifier(1.0 * RBF(1.0)),
    "DT": DecisionTreeClassifier(max_depth = 5),
    "MLP": MLPClassifier(alpha = 0.1, max_iter = 1000),
    "Bayes": GaussianNB()
}

x, y = make_classification(n_features = 2, n_redundant = 0, n_informative = 2, n_clusters_per_class = 1)

rng = np.random.RandomState(2)
x += 1 * rng.uniform(size = x.shape)
linearly_separable = (x, y)

datasets = [make_moons(noise = 0.1), make_circles(noise = 0.1, factor = 0.5), linearly_separable]

cm = plt.cm.RdBu
cm_bright = ListedColormap(["#FF0000", "#0000FF"])

###############################################################################

model_name = "Bayes" # Se agrega aqui el tipo de modelo a ejecutar.

figure = plt.figure(figsize = (9, 3))
h = .02 # Step
i = 1 # Counter

# Iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    x, y = ds
    x = StandardScaler().fit_transform(x)
    
    # Train and test
    xtrain, xtest, ytrain, ytest = train_test_split(x, y)
    
    # Min and Max for normalize data.
    x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
    y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Classifications
    model = classifiers[model_name]
    ax = plt.subplot(1, 3, i)
    
    # Training
    model.fit(xtrain, ytrain)
    score_train = model.score(xtrain, ytrain)
    score_test = model.score(xtest, ytest)
    
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    
    if hasattr(model, "decision_function"):
        zz = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        zz = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        
    # Put the result into a color plot
    zz = zz.reshape(xx.shape)
    ax.contourf(xx, yy, zz, cmap = cm, alpha = .8)
    
    # Plot the training points
    ax.scatter(xtrain[:, 0], xtrain[:, 1], c = ytrain, cmap = cm_bright, edgecolors = "k", alpha = 0.6)
    
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    
    ax.text(xx.max() - .3, yy.min() + .7, "%.2f" % score_train, size = 15, horizontalalignment = "right")
    ax.text(xx.max() - .3, yy.min() + .3, "%.2f" % score_test, size = 15, horizontalalignment = "right")
    
    i += 1
    
    plt.tight_layout()
    # plt.show()
    # plt.savefig("Bayes.eps", format="eps")