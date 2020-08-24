# this code runs perfectly when i remove standard scalr and train_test_split but when i add them it gives me errors :( Still figuring them out tho.


import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters

    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional
    Returns

    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy



def plot_contours(ax, clf, xx, yy, **params):

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

# import some data to play with
iris = datasets.load_iris()
# Take the first two features. We could avoid this by using a
# two-dim dataset
X = iris.data[:, :2]
y = iris.target

# pre-processing
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)


models = (KNeighborsClassifier(n_neighbors=6, p=2, metric='minkowski'),
          KNeighborsClassifier(n_neighbors=30, p=2, metric='minkowski'),
          KNeighborsClassifier(n_neighbors=4, p=2, metric='minkowski'),
          KNeighborsClassifier(n_neighbors=3, p=2, metric='minkowski'),
          KNeighborsClassifier(n_neighbors=2, p=2, metric='minkowski'),
          KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski'))

models = (clf.fit(X_train, y_train) for clf in models)  # sklearn loop over the models

# title for the plots
titles = ('KNN 6',
          'KNN 30',
          'KNN 4',
          'KNN 3',
          'KNN 2',
          'KNN 1')

fig, sub = plt.subplots(2, 2, figsize=(12, 15))
plt.subplots_adjust(wspace=0.2, hspace=0.4)

X0, X1 = X_train[:, 0], X_train[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    #X_LVQ = clf.weights
    #y_LVQ = clf.label_weights
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)

    ax.scatter(X0, X1, c=y_train, cmap=plt.cm.coolwarm, s=20, edgecolors='k')

    # ax.scatter(X_LVQ[:, 0], X_LVQ[:, 1], c=y_LVQ,
    #            cmap=plt.cm.coolwarm, s=50, marker='^', edgecolors='k')

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
plt.show()
