'''Iris Classification Model: Machine learning model that will allow us to Classify species of iris flowers.
Use case: Hamoud wants to determine the species of an iris flower based on characteristics of that flower. For instance attributes including petal
length, width, etc. are the "features" that determine the classification of a given iris flower.
Goal: Use different classification models to classify a given iris sample
Iris setosa, Iris virginica and Iris versicolor.
There are number of models will be used:
1- KNN1 and KNN5
2- Decision Tree


'''

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

# load dataset
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]  # extract only the lengths and widthes of the petals
y = iris.target

''' In order to determine how well our model performs, we need to run it on data it has not seen before, that is, we need to run it on a new set of
measurements and see where our model categorizes this new item. To do this, we can split our data up into two sets; a training and testing
The training set will be what our model uses to learn
The testing set will be the remaining set that assesses whether the model is able to accurately predict the outcome of the measurements from this set.
I will be using a 105/45 split for train/test respectively. That is, we will be training our model on 70% of our data, and then testing on the remaining
30%. A 105/45 split is a reasonable rule to use as a starting point.
Split our dataset into training and testing sets.

random_state takes the data we give it and the target we give it and it is going randomly split them up into a train_test_split. It is going to do that randomly everytime the program runs
if we run the program with random_state equals to zero we are going to get the same exact components from this particular dataset in these variables set.

test_size is set like this because the parameter defines the amount of data used in the test split. since the test_size is 0.3 then train_size will be 0.7
When you’re working with a learning model, it is important to scale the features to a range which is centered around zero.

This is done so that the variance of the features are in the same range.
If a feature’s variance is orders of magnitude more than the variance of other features,
that particular feature might dominate other features in the dataset, which is not something we want happening in our model. '''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

sc = StandardScaler()

sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

print('There are {} samples in the training set and {} samples in the test set'.format(
    X_train.shape[0], X_test.shape[0]))

markers = ('s', 'x', 'o')
colors = ('red', 'blue', 'lightgreen')
cmap = ListedColormap(colors[:len(np.unique(y_test))])
for idx, cl in enumerate(np.unique(y)):
    plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], facecolors=[cmap(idx)], marker=markers[idx], label=cl)


    # Plotting decision regions
    def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

        # setup marker generator and color map
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])

        # plot the decision surface
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution))
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        # plot class samples
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                        alpha=0.8, facecolors=cmap(idx),
                        marker=markers[idx], label=cl)

        # highlight test samples
        if test_idx:
            X_test, y_test = X[test_idx, :], y[test_idx]
            plt.scatter(X_test[:, 0], X_test[:, 1], facecolors='None',
                        alpha=1.0, linewidth=1, marker='o',
                        s=55, label="test set", edgecolors='k')

# create now a Perceptron and fit the data X and y
ppn = Perceptron(n_iter_no_change=40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)

# now we are ready for predictions
y_pred = ppn.predict(X_test_std)
print('Misclassfied samples: %d' % (
            y_test != y_pred).sum())  # refers to the number of individual that we know that below a category that are classified by the method in a different category.
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

# If we plot the original data, we can see that one of the classes is linearly separable, but the other two are not.
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=ppn, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.title('Original Dataset')
plt.show()

#the default metric is minkowski and with p=2 is equivalent to the standard Euclidean metric.
knn = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')
knn.fit(X_train_std, y_train)
print('The accuracy of the knn 1 classifier is {:.2f} out of 1 on training data'.format(knn.score(X_train_std, y_train)))
print('The accuracy of the knn 1 classifier is {:.2f} out of 1 on test data'.format(knn.score(X_test_std, y_test)))
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=knn, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.title('KNN 1')
plt.show()

# we change KNN to a higher value to experiment what we get
knn5 = KNeighborsClassifier(n_neighbors=100, p=2, metric='minkowski')
knn5.fit(X_train_std, y_train)
print('The accuracy of the knn 5 classifier is {:.2f} out of 1 on training data'.format(knn5.score(X_train_std, y_train)))
print('The accuracy of the knn 5 classifier is {:.2f} out of 1 on test data'.format(knn5.score(X_test_std, y_test)))
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=knn5, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.title('KNN 5')
plt.show()

# Decision Tree
decision_tree = DecisionTreeClassifier(random_state=0, max_depth=135)
decision_tree.fit(X_train_std, y_train)
print('The accuracy of the decision tree classifier on training data is {:.2f} out of 1'.format(
    decision_tree.score(X_train_std, y_train)))
print('The accuracy of the decision tree classifier on test data is {:.2f} out of 1'.format(
    decision_tree.score(X_test_std, y_test)))
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=decision_tree, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.title('Decision Tree')
plt.show()


