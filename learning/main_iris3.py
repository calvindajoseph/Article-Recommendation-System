# plotting the Iris dataset using KNeighborsClassifier as a test. If it works perfectly then I will add other classifiers

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# load the iris dataset
iris = load_iris()

# save 'features' and 'targets' in X and y respectively
X, y = iris.data, iris.target

# split data into 'test' and 'train' data
train_X, test_X, train_y, test_y = train_test_split(X, y,
        test_size=.2,
        random_state=23,
        stratify=y
    )

# select classifier
cls = KNeighborsClassifier(5)
cls.fit(train_X, train_y)

# predict the 'target' for 'test data'
pred_y = cls.predict(test_X)
test_accuracy = accuracy_score(test_y, pred_y)
print("Accuracy for test data:", test_accuracy)

incorrect_idx = np.where(pred_y != test_y)
print('Wrongly detected samples:', incorrect_idx[0])

# scatter plot to show correct and incorrect prediction
# plot scatter plot : sepal-width vs all features
colors = ['blue', 'orange', 'green']
feature_x= 1 # sepal width
for feature_y in range(iris.data.shape[1]):
    plt.subplot(2, 2, feature_y+1) # subplot starts from 1 (not 0)
    for i, color in enumerate(colors):
        # indices for each target i.e. 0, 1 & 2
        idx = np.where(test_y == i)[0]
        # find the label and plot the corresponding data
        plt.scatter(test_X[idx, feature_x],
                    test_X[idx, feature_y],
                    label=iris.target_names[i],
                    alpha = 0.6, # transparency
                    color=color
                    )

    # overwrite the test-data with red-color for wrong prediction
    plt.scatter(test_X[incorrect_idx, feature_x],
            test_X[incorrect_idx, feature_y],
            color="red",
            marker='^',
            alpha=0.5,
            label="Incorrect detection",
            s=120 # size of marker
            )

    plt.xlabel('{0}'.format(iris.feature_names[feature_x]))
    plt.ylabel('{0}'.format(iris.feature_names[feature_y]))
    plt.legend()
plt.show()