# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np


boston = datasets.load_boston()
x = boston.data
y = boston.target



def LR_learn(x, y, iter_max, eps):
    w = np.zeros(x.shape[1])
    t = 0
    old_error = 10**4
    diff_error = 10*4
    
    while (t < iter_max and (diff_error > eps)):
        h = np.dot(x, w)
        r = y - h
        f = r**2
        error = 0.5 * np.mean(f)
        grad_f = -2 * np.multiply(x.transpose(), r)
        grad_error = 0.5 * np.mean(grad_f, axis = 1)
       # printf((grad_error.shape))
        w = w - eps * grad_error
        diff_error = np.abs(error - old_error)
        old_error = error
        t = t + 1
    return w

def LR_predict(x, w):
    predicted = np.dot(x, w)
    return predicted


eps = 0.0000001
#eps = pas de la fonction
iter_max = 1000
eps = 10**(-5)
w = LR_learn(x, y, iter_max, eps)

predicted = LR_predict(x, w)


w = np.array
t = 0


fig, ax = plt.subplots()
ax.scatter(y, predicted, edgecolors = (0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k --', lw = 4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()