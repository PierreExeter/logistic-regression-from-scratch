'''
Logistic regression algorithm written from scratch using Numpy.
The score of the algorithm is compared against the Sklearn 
implementation for a classic binary classification problem.
Further steps could be to add L2 regularization and multiclass classification.

Inspired from 
https://www.coursera.org/learn/machine-learning
https://github.com/martinpella/logistic-reg
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# CREATE DATASET
X, y = make_classification(
        n_samples=100, 
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=1, 
        n_clusters_per_class=1)
print(X.shape)
print(y.shape)

def add_intercept(X):
    intercept = np.ones((X.shape[0], 1))
    return np.concatenate((intercept, X), axis=1)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def calc_h(X, theta):
    z = np.dot(X, theta)
    h = sigmoid(z)
    return h

# DEFINE HYPERPARAMETERS
lr = 0.01
num_iter = 100000

# FIT MODEL = FIND THE COEFFICIENTS THETA
XX = add_intercept(X)
theta = np.zeros(XX.shape[1])
m = y.size

for i in range(num_iter):
    h = calc_h(XX, theta)
    gradient = np.dot(XX.T, (h - y)) / m
    theta -= lr * gradient  # gradient descent
    
    z = np.dot(XX, theta)
    h = sigmoid(z)
    cost = (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
        
    if i % 10000 == 0:
        print('Cost: {}'.format(cost))

print('Coefficient: {}'.format(theta))

# MAKE PREDICTIONS = USE THE COEFFICIENTS THETA TO ESTIMATE THE PREDICTION PROBABILITIES
preds_prob = calc_h(XX, theta)
preds = preds_prob.round()
print(preds)
print('Score Numpy: {}'.format((preds == y).mean()))

# PLOT DATASET AND DECISION BOUNDARY
plt.figure(figsize=(10, 6))

# define 2d grid
x1_min, x1_max = X[:,0].min(), X[:,0].max(),
x2_min, x2_max = X[:,1].min(), X[:,1].max(),
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
grid = np.c_[xx1.ravel(), xx2.ravel()]

# make predictions on the grid
grid = add_intercept(grid)
probs = calc_h(grid, theta)
probs = probs.reshape(xx1.shape)

# plot contours
ax = plt.gca()
plt.contourf(xx1, xx2, probs, levels=25, cmap=plt.cm.Spectral, alpha=0.8)
plt.contour(xx1, xx2, probs, [0.5], linewidth=2, colors='black') # decision boundary at 0.5
plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), s=40, cmap=plt.cm.Spectral, edgecolors='black')

# save figure
plt.xlabel("$X_1$")
plt.ylabel("$X_2$")
ax.set_xlim([x1_min, x1_max])
ax.set_ylim([x2_min, x2_max])
plt.tight_layout()
plt.savefig('images/decision_boundary.png')
plt.show()

# SKLEARN IMPLEMENTATION
model = LogisticRegression(C=1e20, solver='lbfgs')
model.fit(X, y)
preds = model.predict(X)

print('Score Sklearn: {}'.format((preds == y).mean()))
print(model.intercept_, model.coef_)

